# %% Imports
import os
import torch
import wandb
from utils_train import Trainer
from utils_load_data import load_res_data, load_embeds, load_paragraphs

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)

# %% Sweep Configuration
k = 1024
sweep_config = {
    'method': 'random',  # Options: grid, random, bayes
    'metric': {
        'name': 'val_loss',
        'goal': 'minimize'
    },
    'parameters': {
        'model_type': {
            'values': ['linear', 'mlp']
        },
        'lr': {
            'distribution': 'log_uniform',
            'min': -16.118,  # ln(1e-7)
            'max': -9.21     # ln(1e-4)
        },
        'weight_decay': {
            'distribution': 'log_uniform',
            'min': -16.118,
            'max': -9.21
        },
        'batch_size': {
            'values': [16, 32, 64, 128, 256, 512, 1024, 2*k]
        },
        'groups_to_load': {
            'values': [1,2,3,4,5,6]
        },
        'dropout': {
            'values': [0.0, 0.05, 0.1, 0.2]
        },
        'd_mlp': {
            'values': [k, 2*k, 4*k, 8*k, 12*k, 16*k ]
        },
    },
}

# %% Core Training Function
def train_sweep():
    # Initialize WandB with sweep config
    wandb.init()

    # Build config from sweep parameters
    c = wandb.config
    config = {
        'model_type': c.model_type,
        'batch_size': c.batch_size,
        'group_size': 4,
        'groups_to_load': c.groups_to_load,
        'lr': c.lr,
        'weight_decay': c.weight_decay,
        'dropout': c.dropout,
        'd_mlp': c.d_mlp,
        'num_epochs': 3,  # Fixed for all runs
        'num_files': 99,
        'd_sonar': 1024  # Should match your embeddings
    }

    # Create and run trainer
    trainer = Trainer(config, DEVICE)
    model = trainer.train()

    # Save model with sweep metadata
    filename = f"./checkpoints/sweeps/{wandb.run.id}_{wandb.config.model_type}.pkl"
    trainer.save_checkpoint(filename)

# %% Main Execution
if __name__ == "__main__":
    # Start sweep or normal training
    sweep_id = wandb.sweep(sweep_config, project="sonar-sweeps")
    print(sweep_id)

    # To run the sweep agent:
    wandb.agent(sweep_id, function=train_sweep, count=20)

    # Alternatively, for single run without sweep:
    # train_sweep()
