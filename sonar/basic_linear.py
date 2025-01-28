# %%
import torch

from utils_load_data import load_res_data, load_embeds, load_paragraphs
from utils_welford   import load_or_compute_welford_stats
from utils_train     import Trainer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)

# %% Training Configuration
config = {
    'model_type': 'linear',  # 'mlp' or 'linear'
    #'d_mlp': 1024 * 4,
    'batch_size': 512,
    'num_epochs': 10,
    'num_files': 99,
    'group_size': 4,
    'groups_to_load': 2,
    'lr': 1e-4,
    'weight_decay': 1e-5,
    'dropout': 0.1
}
res_data   = load_res_data(0, groups_to_load=config['groups_to_load'])
embeds     = load_embeds(0)
paragraphs = load_paragraphs()
config['d_res']   = res_data.shape[-1]
config['d_sonar'] = embeds.shape[-1]
c = config

# %% Welford Data Normalization


# Create and run trainer
trainer = Trainer(config, DEVICE)
model = trainer.train()

# Save checkpoint
filename = f"./checkpoints/{c['model_type']}_{c['num_files']}_{c['groups_to_load']}_{c['d_mlp']}.pkl"
trainer.save_checkpoint(filename)

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
        predicted_embed = norm_emb.restore(model(test_input))
        decoded_text = vec2text_model.predict(
            torch.cat([orig_emb, predicted_embed], dim=0),
            target_lang="eng_Latn"
        )
        print(f"### EXAMPLE {index} ###")
        print({"ORIG": decoded_text[0]})
        print({"NEW":  decoded_text[1]})

# %%
