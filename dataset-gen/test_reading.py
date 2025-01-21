import json
import re
from collections import Counter

# meta-llama/Llama-3.2-3B-Instruct
filename = "rp_outputs_3b_003.jsonl"
with open(filename, 'r') as f:
    for line in f:
        data = json.loads(line)
        prompt = data.get('prompt', '')  # Get the prompt
        completion = data.get('completion', '')
        all_data = [prompt, *completion.split("\n\n")]
        print(json.dumps(all_data, indent=4))

        break
        # Replace multiple consecutive double newlines with a single double newline
        cleaned_completion = re.sub(r'\n{2,}', '\n\n', completion)
        paragraphs = cleaned_completion.split('\n\n')
        num_paragraphs = len(paragraphs)
        paragraph_counts.append(num_paragraphs)


