import circuitsvis as cv
import numpy as np
import torch
from taker import Model
from taker.hooks import HookConfig
from datetime import datetime
import json
from os import listdir
from os.path import exists
from tqdm import tqdm
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
import csv
from huggingface_hub import InferenceClient
import sys, os

# Constants
MAX_NEW_TOKENS = 200
TEMPERATURE = 0.3
MODEL = "google/gemma-2-9b-it"
FILENAME = "gemma9b_results/latest_orig_generation.jsonl"
NEUTRAL_FILENAME = "gemma9b_results/latest_neutral_generation.jsonl"

# Initialize InferenceClient
hf_token = os.getenv('HF_TOKEN')  # or hf_token = "YOUR_TOKEN_HERE"
client = InferenceClient(MODEL, token=hf_token)

# Load prompts
with open('/workspace/SPAR/interp-ab/promptsV1.csv', newline='') as f:
    reader = csv.reader(f)
    readdata = list(reader)[:20]
    print(f"Number of prompts: {len(readdata)}")

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def get_completion(prompt):
    message = client.text_generation(
        prompt,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        return_full_text=False,
        stream=False,
    )
    return message

# Function to write generations to file
def write_generation(filename, data):
    if not exists(filename):
        with open(filename, "w") as f:
            pass
    with open(filename, "a") as file:
        json.dump(data, file)
        file.write("\n")

# Generate outputs for each prompt
for prompt in tqdm(readdata):
    prompt = prompt[0]
    full_prompt = f"Human: {prompt}\n\nAssistant: "

    with HiddenPrints():
        for _ in tqdm(range(50)):
            output = get_completion(full_prompt)

            data = {
                "temperature": TEMPERATURE,
                "max_new_tokens": MAX_NEW_TOKENS,
                "model": MODEL,
                "type": "original",
                "transplant_layers": None,
                "prompt": prompt,
                "output": output,
            }

            write_generation(FILENAME, data)

# Generate neutral outputs
neutral_prompts = ["\n\n"]
for neutral in tqdm(neutral_prompts):
    full_neutral_prompt = f"Human: {neutral}\n\nAssistant: "

    with HiddenPrints():
        for _ in tqdm(range(50)):
            output = get_completion(full_neutral_prompt)

            data = {
                "temperature": TEMPERATURE,
                "max_new_tokens": MAX_NEW_TOKENS,
                "model": MODEL,
                "type": "neutral",
                "transplant_layers": None,
                "prompt": neutral,
                "output": output,
            }

            write_generation(NEUTRAL_FILENAME, data)

print(f"Generations saved to {FILENAME} and {NEUTRAL_FILENAME}")