import textwrap
from termcolor import colored
import argparse
import json
import torch
import einops
from tqdm import tqdm
import numpy as np
from types import SimpleNamespace
from typing import List, Tuple

import utils_load_data
from taker import Model
from transformers import AutoTokenizer

torch.set_grad_enabled(False)

# %%
LIMIT = 100

# DEFINE CODE FOR BATCHED GENERATION

def generate_batch_fast(m, input_ids, attn_masks, max_new_tokens, temperature, exclude_tokens=0) -> Tuple[List[str], List[str]]:
    if m.tokenizer.pad_token is not None:
        pass
    elif m.tokenizer.eos_token is not None:
        m.tokenizer.pad_token = m.tokenizer.eos_token
    else:
        raise ValueError("Tokenizer has neither pad_token nor eos_token defined.")

    orig_len = input_ids.shape[1]
    # Generate outputs
    generate_ids = m.predictor.generate(
        input_ids=input_ids.to(m.device),
        attention_mask=attn_masks.to(m.device),
        max_length=input_ids.shape[1] + max_new_tokens,
        do_sample=True,
        temperature=temperature,
        pad_token_id=m.tokenizer.pad_token_id,
    )
    # Decode all generated sequences at once
    batch_text_after = m.tokenizer.batch_decode(
        [ids[orig_len:] for ids in generate_ids],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )

    # Get original prompts
    batch_prompts = m.tokenizer.batch_decode(
        input_ids[:, exclude_tokens:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )

    return batch_prompts, batch_text_after


def move_pad_tokens_to_left(token_ids, attn_mask, pad_token_id):
    # If there are any tokens that are m.tokenizer.pad_token_id, we need to swap
    # them to the left.
    # i.e: ids:  [a, b, c, d, 0, 0, 0] -> [0, 0, 0, a, b, c, d]
    # i.e: mask: [1, 1, 1, 1, 0, 0 ,0] -> [0, 0, 0, 1, 1, 1, 1]
    # Find where the padding tokens are
    pad_mask = token_ids == pad_token_id
    total_tokens = token_ids.shape[1]
    token_ids = token_ids.copy()
    attn_mask = attn_mask.copy()

    # For each row, count number of non-pad tokens and create new indices
    for i in range(len(token_ids)):
        non_pad_count = np.sum(~pad_mask[i])
        if non_pad_count < total_tokens:
            # Number of padding tokens needed
            n_pad = total_tokens - non_pad_count

            # Shift non-pad tokens to the right
            token_ids[i] = np.concatenate([
                np.full(n_pad, pad_token_id),
                token_ids[i][pad_mask[i] == 0]
            ])

            # Also shift attention mask
            attn_mask[i] = np.concatenate([
                np.zeros(n_pad),
                attn_mask[i][pad_mask[i] == 0]
            ])

    return token_ids, attn_mask


# %%
def main(verbose=False):
    args = SimpleNamespace(
        res_index=99,
        max_new_tokens=128,
        temperature=0.3
    )

    m = Model("meta-llama/Llama-3.2-3B-Instruct", dtype="bfp16", add_hooks=False)
    m.tokenizer.pad_token_id = m.tokenizer.eos_token_id
    m.show_details()


    # Load paragraphs
    embeds = utils_load_data.load_embeds(99)
    n_paragraphs = embeds.shape[0]
    contexts = utils_load_data.load_full_contexts()
    contexts = contexts[-n_paragraphs:]
    n_texts = len(contexts)

    print(len(contexts), type(contexts), type(contexts[0]))
    print(contexts[0])
    print(contexts[1])
    print(contexts[2])

    # Encode paragraphs
    context_tokens = m.tokenizer(contexts, padding=True, truncation=False, max_length=1000, return_tensors="pt")
    token_ids = context_tokens.input_ids
    attn_mask = context_tokens.attention_mask

    batch_size = 5
    outputs = []
    for i in tqdm(range(0, n_texts, batch_size)):
        prompt = torch.tensor(token_ids[i:i+batch_size], dtype=torch.long, device=m.device)
        masks  = torch.tensor(attn_mask[i:i+batch_size], dtype=torch.long, device=m.device)
        _prompts, texts = generate_batch_fast(m, prompt, masks, args.max_new_tokens, args.temperature, exclude_tokens=0)
        texts = [t.split('\n\n')[0] for t in texts]
        outputs.extend(texts)

        if LIMIT and i+batch_size >= LIMIT:
            break

    result_data = {
        "model": m.model_repo,
        "prompt": "\n\n",
        "res_index": args.res_index,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "outputs": outputs,
    }
    out_filename = f"./comparison_texts/regenerated_outputs.json"
    with open(out_filename, "w") as f:
        f.write(json.dumps(result_data, indent=2) + "\n")
    print(f"Result written to {out_filename}")

if __name__ == "__main__":
    main()

