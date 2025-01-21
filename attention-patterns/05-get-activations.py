import json
from taker import Model

m = Model("google/gemma-2-9b-it", dtype="hqq8", compile=True, device_map="cuda")

# load generations from /gemma9b_results/latest_neutral_generation.jsonl 
with open("./gemma9b_results/latest_neutral_generation.jsonl", "r") as file:
    generations = [json.loads(line) for line in file]

# get the activations for the generations
activations = []
for generation in generations:
    activations.append(m.get_attn_weights(generation["full_text"]))

print(activations)


