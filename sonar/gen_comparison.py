# # %%
# import json
# from datetime import datetime
# from os.path import exists
# from typing import List, Tuple

# import pandas as pd
# import torch
# from taker import Model

# # set torch to use inference mode globally
# torch.set_grad_enabled(False)

# # %%
# m = Model("meta-llama/Llama-3.1-8B-Instruct", dtype="bfp16")
# m.show_details()

# # %%
# folder = f"../data/llama8b"
# prefix = "V2"
# current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
# max_new_tokens = 100
# temperature = 0.3
# neutral_prompt = "\n\n"
# batch_size = 10

# # %%

# # DEFINE CODE FOR BATCHED GENERATION
# from transformers import AutoTokenizer
# m.tokenizer = AutoTokenizer.from_pretrained(m.tokenizer_repo, legacy=False, padding_side='left')
# if m.tokenizer.pad_token is not None:
#     pass
# elif m.tokenizer.eos_token is not None:
#     m.tokenizer.pad_token = m.tokenizer.eos_token
# else:
#     raise ValueError("Tokenizer has neither pad_token nor eos_token defined.")

# def generate_batch_fast(m, batch_prompts, max_new_tokens, temperature) -> Tuple[List[str], List[str]]:
#     # Tokenize all prompts in the batch
#     batch_encodings = m.tokenizer(
#         batch_prompts,
#         padding=True,
#         truncation=False,
#         max_length=1000,
#         return_tensors="pt"
#     )
#     orig_len = batch_encodings.input_ids.shape[1]
#     # Generate outputs
#     generate_ids = m.predictor.generate(
#         input_ids=batch_encodings.input_ids.to(m.device),
#         attention_mask=batch_encodings.attention_mask.to(m.device),
#         max_length=batch_encodings.input_ids.shape[1] + max_new_tokens,
#         do_sample=True,
#         temperature=temperature,
#         pad_token_id=m.tokenizer.pad_token_id,
#     )
#     # Decode all generated sequences at once
#     batch_text_after = m.tokenizer.batch_decode(
#         [ids[orig_len:] for ids in generate_ids],
#         skip_special_tokens=True,
#         clean_up_tokenization_spaces=False
#     )

#     return batch_prompts, batch_text_after


# transfer_scale = 1
# transfer_tokens = 1
# neutral_prompt = "\n\n"

# filename = f"{folder}/{prefix}_transferred_{transfer_tokens}t_{transfer_scale}x_generation.jsonl"
# if not exists(filename):
#     with open(filename, "w") as f:
#         pass

# for info_prompt in orig_df['part1']:
#     #RESET HOOKS BEFORE TRANSPLANTING NEXT SET OF ACTIVATIONS
#     [h.reset() for h in m.hooks.neuron_replace.values()]
#     acts = m.get_midlayer_activations(info_prompt)
#     orig_token_index = m.get_ids(info_prompt).shape[1] - 1
#     new_token_index  = m.get_ids(neutral_prompt).shape[1] - 1

#     for layer_index in range(0,m.cfg.n_layers):
#         m.hooks.neuron_replace[f"layer_{layer_index}_mlp_pre_out"].add_token(new_token_index, acts["mlp"][0, layer_index, orig_token_index]*transfer_scale)
#         m.hooks.neuron_replace[f"layer_{layer_index}_attn_pre_out"].add_token(new_token_index, acts["attn"][0, layer_index, orig_token_index]*transfer_scale)

#     output = m.generate(neutral_prompt, max_new_tokens, temperature=temperature)

#     data = {
#         "temperature": temperature,
#         "max_new_tokens": max_new_tokens,
#         "model": m.model_repo,
#         "type": "transferred",
#         "num_transferred_tokens": transfer_tokens,
#         "transfer_scale": transfer_scale,
#         "transplant_layers": (0,m.cfg.n_layers-1),
#         "orig_prompt": info_prompt,
#         "transplant_prompt": neutral_prompt,
#         "output": output[1],
#     }

#     with open(filename, "a") as file:
#         file.write(json.dumps(data) + "\n")


import utils_load_data

def load_res_data(index, group_size=4, groups_to_load=2):
    file_path = f"./tensors/res_data_{index:03d}.pt"
    data = torch.load(file_path, map_location="cpu", weights_only=False)
    data = torch.cat(data, dim=2)  # Concatenate list of tensors
    data = data.squeeze(0)
    data = data[1:, :, :]  # Remove first layer
    data = einops.rearrange(data, 'layers samples dim -> samples layers dim')
    data = einops.rearrange(data, 'samples (layersg g) dim -> samples layersg (g dim)', g=group_size)
    data = einops.rearrange(data[:, -groups_to_load:], 'samples layersg gdim -> samples (layersg gdim)')
    return data.float()

res_data  = utils_load_data.load_res_data(99, groups_to_load=14)