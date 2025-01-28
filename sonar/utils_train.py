import os
import gc
import pickle
from tqdm import tqdm
import torch
import wandb
from types import SimpleNamespace
from torch.utils.data import DataLoader, TensorDataset
from utils_models  import Linear, MLP
from utils_welford import load_or_compute_welford_stats, Normalizer
from utils_load_data import load_res_data, load_embeds

class Trainer:
    def __init__(self, config, device):
        self._config = config
        self.device = device

        welford_data = load_or_compute_welford_stats(self.c.groups_to_load)
        self.normalizer_emb = welford_data.norm_emb
        self.normalizer_res = welford_data.norm_res

        d_res = load_res_data(0, self.c.group_size, self.c.groups_to_load).shape[-1]
        self._config["d_res"] = d_res

        # Initialize model
        self.model = self._init_model().to(device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.c.lr,
            weight_decay=self.c.weight_decay
        )
        self.criterion = torch.nn.MSELoss()

    @property
    def c(self):
        return SimpleNamespace(**self._config)

    def _init_model(self):
        if self.c.model_type == 'linear':
            return Linear(self.c)
        elif self.c.model_type == 'mlp':
            return MLP(self.c)
        else:
            raise ValueError(f"Unknown model type: {self.c.model_type}")

    def train_epoch(self, epoch):
        self.model.train()
        train_loss = 0
        train_batches = 0

        for file_idx in (pbar := tqdm(range(self.c.num_files), desc=f"Epoch {epoch+1}")):
            res_data = load_res_data(file_idx, groups_to_load=self.c.groups_to_load)
            embeds = load_embeds(file_idx)
            dataset = TensorDataset(res_data, embeds)
            train_loader = DataLoader(dataset, batch_size=self.c.batch_size, shuffle=True)

            for x, y in train_loader:
                x = self.normalizer_res.normalize(x.to(self.device))
                y = self.normalizer_emb.normalize(y.to(self.device))

                self.optimizer.zero_grad()
                outputs = self.model(x)
                loss = self.criterion(outputs, y)
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()
                train_batches += 1
                pbar.set_postfix({"Loss": f"{train_loss/train_batches:.4f}"})

            del res_data, embeds, dataset
            gc.collect()
            torch.cuda.empty_cache()

        return train_loss / train_batches

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        val_loss = 0
        VALIDATION_FILE_INDEX = 99
        with torch.no_grad():
            res_data = load_res_data(VALIDATION_FILE_INDEX, groups_to_load=self.c.groups_to_load)
            embeds = load_embeds(VALIDATION_FILE_INDEX)
            dataset = TensorDataset(res_data, embeds)
            test_loader = DataLoader(dataset, batch_size=self.c.batch_size)

            for x, y in test_loader:
                x = self.normalizer_res.normalize(x.to(self.device))
                y = self.normalizer_emb.normalize(y.to(self.device))
                outputs = self.model(x)
                val_loss += self.criterion(outputs, y).item()

            del res_data, embeds, dataset
            gc.collect()
            torch.cuda.empty_cache()

        return val_loss / len(test_loader)

    def train(self):
        torch.set_grad_enabled(True)

        for epoch in range(self.c.num_epochs):
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate()  # Validate on next file

            wandb.log({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
            })

            print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        return self.model

    def save_checkpoint(self, filename):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'wb') as f:
            pickle.dump({
                'model': self.model.state_dict(),
                'config': self.c,
                'welford_emb': self.normalizer_emb.welford,
                'welford_res': self.normalizer_res.welford,
            }, f)

    @classmethod
    def load_checkpoint(cls, filename, device):
        with open(filename, 'rb') as f:
            checkpoint = pickle.load(f)

        config = checkpoint['config']
        normalizer_emb = Normalizer(checkpoint['welford_emb'])
        normalizer_res = Normalizer(checkpoint['welford_res'])

        trainer = cls(config, normalizer_emb, normalizer_res, device)
        trainer.model.load_state_dict(checkpoint['model'])
        return trainer