# %%
import pickle
import os
import gc
from time import time
import json
from tqdm import tqdm
import torch
import einops
import torch.nn as nn
from welford_torch import Welford
from torch.utils.data import DataLoader, TensorDataset
import wandb

from load_data import load_res_data, load_embeds, load_paragraphs

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)

embeds     = load_embeds(0)
res_data   = load_res_data(0, groups_to_load=3)
paragraphs = load_paragraphs()

print("embeds shape  : ", embeds.shape)
print("res data shape: ", res_data.shape)
print("paragraphs    : ", len(paragraphs))

D_RES = res_data.shape[-1]
D_SONAR = embeds.shape[-1]

# %%

# Try to load Welford statistics from pickle file, compute if not exists
try:
    with open('./welford_data/welford_stats_10.pkl', 'rb') as f:
        print("Loading existing welford data")
        welford_stats = pickle.load(f)
        welford_emb = welford_stats['welford_emb']
        welford_res = welford_stats['welford_res']

except FileNotFoundError:
    welford_emb = Welford()
    welford_res = Welford()

    for i in tqdm(range(10)):
        res_data = load_res_data(i).cuda()
        embeds = load_embeds(i).cuda()

        welford_res.add_all(res_data)
        welford_emb.add_all(embeds)
        del res_data, embeds
        gc.collect()
        torch.cuda.empty_cache()

    # Save Welford statistics for first 10 files using pickle

    os.makedirs('./welford_data', exist_ok=True)
    with open('./welford_data/welford_stats_10.pkl', 'wb') as f:
        pickle.dump({
            'welford_emb': welford_emb,
            'welford_res': welford_res
        }, f)

# %%

def normalize_res(res_data):
    """Normalize residual data using precomputed mean and variance"""
    return (res_data - welford_res.mean) / torch.sqrt(welford_res.var_s + 1e-8)

def normalize_emb(emb_data):
    """Normalize embedding data using precomputed mean and variance"""
    return (emb_data - welford_emb.mean) / torch.sqrt(welford_emb.var_s + 1e-8)

def restore_emb(normed_emb_data):
    """Restore normalized embedding data to original scale using precomputed mean and variance"""
    return normed_emb_data * torch.sqrt(welford_emb.var_s + 1e-8) + welford_emb.mean

# %%
# Train linear model
# Create model

class Linear(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(D_RES, D_SONAR)

    def forward(self, x):
        return self.linear(x)
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        # More balanced hidden layer dimensions based on input/output sizes
        d_hidden1 = 1024*8  # First hidden layer dimension
        d_hidden2 = 1024*8  # Second hidden layer dimension
        self.sequential = nn.Sequential(
            nn.Linear(D_RES, d_hidden1),
            nn.GELU(),
            nn.Dropout(0.1),  # Add dropout for regularization
            nn.Linear(d_hidden1, d_hidden2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_hidden2, D_SONAR)
        )

    def forward(self, x):
        return self.sequential(x)

def train_model(_model):

    wandb.init()
    c = wandb.config
    torch.set_grad_enabled(True)
    _model = ResidualToEmbed().to(DEVICE)
    criterion = nn.MSELoss()

    # Training loop
    c.batch_size = 256  # Reduced batch size for less memory usage
    c.num_epochs = 10     # Increased epochs for better convergence
    c.num_files  = 99      # Increased number of files for training
    c.lr = 1e-4
    c.weight_decay = 1e-5
    # Use weight decay for regularization
    optimizer = torch.optim.Adam(_model.parameters(), lr=c.lr, weight_decay=c.weight_decay)

    for epoch in range(c.num_epochs):
        # Training
        _model.train()
        train_loss = 0
        train_batches = 0
        # Process files in chunks to manage memory
        for file_idx in (pbar := tqdm(range(c.num_files), desc=f"Epoch {epoch+1}")):
            # Load and normalize data for current file
            res_data = load_res_data(file_idx)
            embeds = load_embeds(file_idx)

            dataset = TensorDataset(res_data, embeds)

            # Split into train/val for this file
            train_loader = DataLoader(dataset, batch_size=c.batch_size, shuffle=True)

            # Train on current file
            for x, y in train_loader:
                x, y = normalize_res(x.to(DEVICE)), normalize_emb(y.to(DEVICE))

                optimizer.zero_grad()
                outputs = _model(x)
                loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                train_batches += 1

                # Update tqdm bar with current loss
                pbar.set_postfix({"Loss": f"{train_loss/train_batches:.4f}"})

            # Clean up memory
            del res_data, embeds, dataset
            gc.collect()
            torch.cuda.empty_cache()

        # Validation (using last file's validation set)
        _model.eval()
        val_loss = 0
        with torch.no_grad():
            res_data = load_res_data(file_idx+1)
            embeds = load_embeds(file_idx+1)

            dataset = TensorDataset(res_data, embeds)
            test_loader = DataLoader(dataset, batch_size=c.batch_size)
            for x, y in test_loader:
                x, y = normalize_res(x.to(DEVICE)), normalize_emb(y.to(DEVICE))
                outputs = _model(x)
                val_loss += criterion(outputs, y).item()
            del res_data, embeds, dataset
            gc.collect()
            torch.cuda.empty_cache()

        wandb.log({
            "epoch": epoch+1,
            "train_loss": train_loss / train_batches,
            "val_loss": val_loss / len(test_loader) if train_batches > 0 else float('inf'),
        })

        print(f"Epoch {epoch+1}: Train Loss: {train_loss/train_batches:.4f}, Val Loss: {val_loss/len(test_loader):.4f}")
    return _model

# %%
filename = "./checkpoints/mlp_99_triple.pkl"

ResidualToEmbed = MLP
model = ResidualToEmbed()

if not os.path.exists(filename):
    model = train_model(model)
    # Train and save model if checkpoint doesn't exist
    with open(filename, 'wb') as f:
        pickle.dump({
            'model': model,
            'welford_emb': welford_emb,
            'welford_res': welford_res,
        }, f)
else:
    # Load model if checkpoint exists
    with open(filename, 'rb') as f:
        checkpoint = pickle.load(f)
        model.load_state_dict(checkpoint['model'].state_dict())
        welford_emb = checkpoint['welford_emb']
        welford_res = checkpoint['welford_res']

    model = model.to('cuda')


# %%
# Test outputs
print("Testing some outputs")
from sonar.inference_pipelines.text import EmbeddingToTextModelPipeline
vec2text_model = EmbeddingToTextModelPipeline(decoder="text_sonar_basic_decoder", tokenizer="text_sonar_basic_encoder", device=DEVICE)

# Example usage with the vec2text_model
with torch.no_grad():
    for index in [1, 100, 200, 300, 500, 800, 1000]:
        orig_emb   = embeds[index].unsqueeze(dim=0).to(DEVICE)
        test_input = res_data[index].unsqueeze(dim=0).to(DEVICE)
        predicted_embed = restore_emb(model(test_input))
        decoded_text = vec2text_model.predict(
            torch.cat([orig_emb, predicted_embed], dim=0),
            target_lang="eng_Latn"
        )
        print(f"### EXAMPLE {index} ###")
        print({"ORIG": decoded_text[0]})
        print({"NEW":  decoded_text[1]})

# %%
