import torch
from taker import Model
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import json
from dataclasses import dataclass

# Disable gradients
torch.set_grad_enabled(False)

# Load the model
m = Model("google/gemma-2-9b-it", dtype="hqq8", compile=True, device_map="cuda")
m.show_details()

# Initialize SentenceTransformer
sentenceTransformer = SentenceTransformer("all-mpnet-base-v2")
# sentenceTransformer = SentenceTransformer("dunzhang/stella_en_1.5B_v5")

# Neutral prompt
neutral_prompt = "---\n\n"
# neutral_prompt = "------------------------------------------------\n\n"
tokens_to_transfer = 2
neutral_ids = m.get_ids(neutral_prompt)
neutral_embeds = m.get_inputs_embeds(input_ids=neutral_ids)

# TunableInputsEmbeds class
class TunableInputsEmbeds(torch.nn.Module):
    def __init__(self, inputs_embeds):
        super().__init__()
        self.embeds = torch.nn.Parameter(inputs_embeds)
        self.shape = self.embeds.shape

    def forward(self):
        return self.embeds

def reset_hooks():
    #RESET HOOKS BEFORE TRANSPLANTING NEXT SET OF ACTIVATIONS
    [h.reset() for h in m.hooks.neuron_replace.values()]

def read_prompts(path="./results/latest_phi3_generations.jsonl"):
    with open(path, "r") as file:
        invalid_count = 0
        for line in (pbar:=tqdm(file)):
            data = json.loads(line)
            full_text = data['full_text']

            # Split the full text into prompt and output
            prompt, output = full_text.split("Assistant:", 1)
            prompt += "Assistant:"  # Add back the "Assistant:" part

            # Tokenize the full text and find the start of the output
            full_tokens = m.tokenizer.encode(full_text)
            output_start = len(m.tokenizer.encode(prompt))

            # print("Input, Output Tokens:", output_start, len(full_tokens))

            # Find the index of "\n\n" after 100 tokens into the output
            output_tokens = full_tokens[output_start:]
            if len(output_tokens) > 100:
                text_before_100_tokens = m.tokenizer.decode(output_tokens[:100])
                text_after_100_tokens = m.tokenizer.decode(output_tokens[100:])
                text_after_100_tokens_until_newline = text_after_100_tokens.split("\n\n")[0]

                if text_after_100_tokens_until_newline != text_after_100_tokens:
                    full_index = m.tokenizer.encode(prompt + text_before_100_tokens + text_after_100_tokens_until_newline)
                    data['split_index'] = len(full_index)
                else:
                    data['split_index'] = -1
            else:
                data['split_index'] = -1

            if data['split_index'] == -1:
                invalid_count += 1
                # update pbar with invalid count.
                pbar.set_description(f"Invalid prompts: {invalid_count}")
                continue

            data["newline_index"] = data["split_index"]

            yield data


# Load prompts
prompts = list(read_prompts("./results/latest_phi3_generations.jsonl"))
print(prompts)

new_token_index = m.get_ids(neutral_prompt).shape[1] - 1

def generate_with_transfer(prompt_data, num_tokens=20):
    reset_hooks()
    full_ids = m.get_ids(prompt_data['full_text'])
    orig_newline_index = prompt_data['newline_index']
    ids_prompt = full_ids[:, :orig_newline_index+1]
    neutral_ids = m.get_ids(neutral_prompt)

    # do neutral generation
    _, generated_text_neutral = m.generate(input_ids=neutral_ids, num=num_tokens, temperature=0.3)

    # do original generation
    _, generated_text_orig = m.generate(input_ids=ids_prompt, num=num_tokens, temperature=0.3)

    # Get original text activations
    acts = m.get_midlayer_activations(input_ids=ids_prompt)
    orig_acts = {
        "mlp": acts["mlp"][0, :, orig_newline_index],
        "attn": acts["attn"][0, :, orig_newline_index]
    }
    for j in range(tokens_to_transfer):
        # Transfer activations
        for layer_index in range(m.cfg.n_layers):
            m.hooks.neuron_replace[f"layer_{layer_index}_mlp_pre_out"].add_token(new_token_index-j, orig_acts["mlp"][layer_index])
            m.hooks.neuron_replace[f"layer_{layer_index}_attn_pre_out"].add_token(new_token_index-j, orig_acts["attn"][layer_index])

    # do new transfered generation
    _, generated_text_new = m.generate(input_ids=neutral_ids, num=num_tokens, temperature=0.3)

    return generated_text_orig, generated_text_new, generated_text_neutral

# Generate and embed texts
all_generated = []

for prompt_data in tqdm(prompts[:100]):
    for i in range(3):
        reset_hooks()
        generated_text_orig, generated_text_new, generated_text_neutral = generate_with_transfer(prompt_data, 50)
        text = {"attempt": i, "orig": generated_text_orig, "new": generated_text_new, "neutral": generated_text_neutral, "prompt": prompt_data['full_text']}
        all_generated.append(text)

# save all_generated to jsonl
with open("./results/latest_phi3_generations_gemma2-9b_{tokens_to_transfer}t_x3.jsonl", "w") as file:
    for text in all_generated:
        file.write(json.dumps(text) + "\n")

