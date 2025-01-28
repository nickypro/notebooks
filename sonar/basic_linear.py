# %%
import pickle
import os
import gc
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import wandb

from utils_load_data import load_res_data, load_embeds, load_paragraphs
from utils_welford   import load_or_compute_welford_stats, normalize_emb, normalize_res, restore_emb
from utils_models    import Linear, MLP

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)

GROUPS_TO_LOAD = 2
D_MLP = 1024 * 8
BATCH_SIZE = 512

res_data   = load_res_data(0, groups_to_load=GROUPS_TO_LOAD)
embeds     = load_embeds(0)
paragraphs = load_paragraphs()

print("embeds shape  : ", embeds.shape)
print("res data shape: ", res_data.shape)
print("paragraphs    : ", len(paragraphs))

D_RES = res_data.shape[-1]
D_SONAR = embeds.shape[-1]

# %% # Make sure data is normalized, using Welford Online
welford_emb, welford_res = load_or_compute_welford_stats(GROUPS_TO_LOAD)

# %%
# Train linear model
# Create model


def train_model(_model):

    torch.set_grad_enabled(True)
    wandb.init()
    c = wandb.config

    # Training loop
    c.batch_size = BATCH_SIZE  # Reduced batch size for less memory usage
    c.num_epochs = 10     # Increased epochs for better convergence
    c.num_files  = 99      # Increased number of files for training
    c.groups_to_load = GROUPS_TO_LOAD
    c.lr = 1e-4
    c.weight_decay = 1e-5
    c.d_res   = D_RES
    c.d_mlp   = D_MLP
    c.d_sonar = D_SONAR
    # Use weight decay for regularization

    criterion = nn.MSELoss()
    _model = ResidualToEmbed().to(DEVICE)
    optimizer = torch.optim.Adam(_model.parameters(), lr=c.lr, weight_decay=c.weight_decay)

    for epoch in range(c.num_epochs):
        # Training
        _model.train()
        train_loss = 0
        train_batches = 0
        # Process files in chunks to manage memory
        for file_idx in (pbar := tqdm(range(c.num_files), desc=f"Epoch {epoch+1}")):
            # Load and normalize data for current file
            res_data = load_res_data(file_idx, groups_to_load=c.groups_to_load)
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
            res_data = load_res_data(file_idx+1, groups_to_load=GROUPS_TO_LOAD)
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
filename = f"./checkpoints/mlp_99_{GROUPS_TO_LOAD}_{D_MLP}.pkl"

ResidualToEmbed = MLP
model = ResidualToEmbed()
print(model)

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
