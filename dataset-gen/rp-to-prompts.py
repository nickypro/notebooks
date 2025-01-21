# %%
# Assume openai>=1.0.0
from openai import OpenAI
from dotenv import load_dotenv
import asyncio
import os
import random
import time
from typing import List, Any
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import itertools


# Load token from env file
load_dotenv("./env")
deepinfra_token = os.environ.get("DEEPINFRA_TOKEN")
MODEL_REPO = "meta-llama/Llama-3.2-3B-Instruct"
# MODEL_REPO = "meta-llama/Llama-3.3-70B-Instruct-Turbo"

def retry_with_exponential_backoff(func,
        retries = 20,
        intial_sleep_time: int = 3,
        jitter: bool = True,
        backoff_factor: float = 1.5):
    """
    This is a sneaky function that gets around the "rate limit error" from GPT (GPT has a maximum tokens per min processed, and concurrent processing may exceed this) by retrying the model call that exceeds the limit after a certain amount of time.
    """
    def wrapper(*args, **kwargs):
        sleep_time = intial_sleep_time
        for attempt in range(retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if "rate_limit_exceeded" in str(e):
                    sleep_time *=  backoff_factor * (1 + jitter * random.random())
                    time.sleep(sleep_time)
                else:
                    raise
        raise Exception(f"Maximum retries {retries} exceeded")
    return wrapper

@retry_with_exponential_backoff
def get_llama_completion(prompt: str) -> str:
    # Create an OpenAI client with your deepinfra token and endpoint
    openai = OpenAI(
        api_key=deepinfra_token,
        base_url="https://api.deepinfra.com/v1/openai",
    )

    chat_completion = openai.chat.completions.create(
        model=MODEL_REPO,
        messages=[{"role": "user", "content": prompt}],
    )

    return chat_completion.choices[0].message.content
def process_in_parallel(items: List[Any], process_func, max_workers: int = 25, batch_size: int = 50):
    """
    Process items in parallel batches using ThreadPoolExecutor.

    Args:
        items: List of items to process
        process_func: Function to apply to each item
        max_workers: Maximum number of concurrent workers
        batch_size: Number of items to process in each batch

    Returns:
        List of results in the same order as input items
    """
    results = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Process items in batches
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            futures = [executor.submit(process_func, item) for item in batch]

            # Collect results from this batch
            batch_results = []
            for future in futures:
                try:
                    result = future.result()
                    batch_results.append(result)
                except Exception as e:
                    print(f"Error processing item: {e}")
                    batch_results.append(None)

            results.extend(batch_results)

            # Optional: Add a small delay between batches to avoid rate limiting
            time.sleep(1)

    return results

def get_prompts_parallel(prompts: List[str], max_workers: int = 3, batch_size: int = 10) -> List[str]:
    def get_prompt_for_text(text: str) -> str:
        try:
            return get_llama_completion(text)
        except Exception as e:
            print(f"Error getting completion: {e}")
            return ""

    return process_in_parallel(prompts, get_prompt_for_text, max_workers, batch_size)

# %%

from datasets import load_dataset
dataset = load_dataset("togethercomputer/RedPajama-Data-1T", "common_crawl", streaming=True)["train"]


for data in dataset:
    print(data)
    print(data.keys())
    break

# %%
get_llama_completion("How many weeks in a fiscal year? Is it always the same?")

# %%
def format_question(data):
    # Create a prompt asking to summarize the document while ignoring specific details
    prompt = f"""
Content: {data['text']}

---

REQUEST:
Write a prompt based on the above text, that is a single-paragraph, high-level description. Make the prompt in the format format similar to: "Write a (news feed/chapter/piece/article/wiki entry/...), titled (document name)', which includes (1-2 sentence list of topics to cover, kept very vague). The full piece should be approximately (n-paragraphs or other unit of length)".

Only provide the prompt, do not write anything else."""
    return prompt

def get_final_prompt(data):
    question = format_question(data)
    answer = get_llama_completion(question)
    return answer

# Process items in the dataset

for data in dataset:
    summary = get_final_prompt(data)
    # print(f"\nDocument ID: {data['identifier']}")
    print(f"Generated Prompt: {summary}\n")
    print("-" * 80)
    break
# %%
import json

output_file = "rp_prompts_3b_i.jsonl"
num_existing = sum(1 for _ in open(output_file)) if os.path.exists(output_file) else 0
batch = []
for index, data in tqdm(itertools.islice(enumerate(dataset), num_existing, 100_000), total=100_000-num_existing):
    batch.append(data)
    data["index"] = index
    if len(batch) >= 50:
        prompts = get_prompts_parallel([format_question(d) for d in batch])
        for data_item, prompt in zip(batch, prompts):
            result = {
                "id": data_item["index"],
                "prompt": prompt,
                
            }
            with open(output_file, "a") as f:
                f.write(json.dumps(result) + "\n")
        batch = []

# %%

