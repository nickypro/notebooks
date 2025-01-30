import textwrap
from termcolor import colored
from argparse import ArgumentParser
import torch
from utils_load_data import load_embeds, load_res_data
from utils_train import Trainer  # Assumption: You have a Trainer class with loading functionality

def main():
    parser = ArgumentParser(description='Test the model')
    parser.add_argument('wandb_run_name', type=str,
                       help='Name of the W&B run to load (e.g., northern-sweep-37)')
    args = parser.parse_args()
    model = Trainer.load_from_wandb("sonar-sweeps/"+args.wandb_run_name)
    return model

if __name__ == "__main__":
    trainer = main()
    trainer.model.eval()
    DEVICE = trainer.device

    print("Testing some outputs")
    from sonar.inference_pipelines.text import EmbeddingToTextModelPipeline
    vec2text_model = EmbeddingToTextModelPipeline(decoder="text_sonar_basic_decoder", tokenizer="text_sonar_basic_encoder", device=DEVICE)

    # Example usage with the vec2text_model
    with torch.no_grad():
        embeds = load_embeds(99)
        res_data = load_res_data(99, groups_to_load=trainer.c.groups_to_load)
        for index in [1, 42, 100, 200, 300, 500, 800, 1000, 1234, 2345, 3456]:
            orig_emb   = embeds[index].unsqueeze(dim=0).to(DEVICE)
            test_input = trainer.normalizer_res(res_data[index].unsqueeze(dim=0).to(DEVICE))
            predicted_embed = trainer.normalizer_emb.restore(trainer.model(test_input))
            decoded_text = vec2text_model.predict(
                torch.cat([orig_emb, predicted_embed], dim=0),
                target_lang="eng_Latn"
            )
            print(f"### EXAMPLE {index} ###")

            print(textwrap.fill(colored(f"ORIGINAL: {decoded_text[0][:200]}", 'blue'),
                              width=120,
                              initial_indent='',
                              subsequent_indent=' ' * 10))
            print(textwrap.fill(colored(f"PROBE   : {decoded_text[1][:200]}", 'green'),
                              width=120,
                              initial_indent='',
                              subsequent_indent=' ' * 10))
            print()

# %%
