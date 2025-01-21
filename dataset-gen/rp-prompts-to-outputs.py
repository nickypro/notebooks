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

import json
import os
from typing import Iterator, Dict
import itertools
from tqdm import tqdm

# Reuse your existing get_llama_completion and parallel processing functions
# ... (keep the imports and other functions from the previous file)

def read_jsonl_in_chunks(filename: str, chunk_size: int = 100) -> Iterator[Dict]:
    """Read JSONL file in chunks to avoid loading entire file into memory"""
    chunk = []
    with open(filename, 'r') as f:
        for line in f:
            chunk.append(json.loads(line))
            if len(chunk) >= chunk_size:
                yield chunk
                chunk = []
        if chunk:  # Don't forget the last partial chunk
            yield chunk

def get_next_file_number(base_filename: str = "rp_outputs_3b_") -> int:
    """Find the next available file number"""
    i = 1
    while os.path.exists(f"{base_filename}{i:03d}.jsonl"):
        i += 1
    return i

def get_file_size_mb(filename: str) -> float:
    """Get file size in megabytes"""
    return os.path.getsize(filename) / (1024 * 1024)

def process_prompts():
    input_file = "rp_prompts_3b_i.jsonl"
    base_output_file = "rp_outputs_3b_"
    chunk_size = 50  # Process and write 50 prompts at a time
    max_file_size_mb = 100  # Now used as a guideline, not a strict limit

    current_file_num = get_next_file_number(base_output_file)
    current_output_file = f"{base_output_file}{current_file_num:03d}.jsonl"

    # Read the last entry's id from the current output file
    last_id = None
    if os.path.exists(current_output_file):
        with open(current_output_file, "r") as f:
            for line in f:
                last_entry = json.loads(line)
                last_id = last_entry["id"]  # Get the last entry's id

    # Read prompts and skip the first 'last_id' prompts
    prompts_to_process = []
    with open(input_file, "r") as f:
        for line in f:
            prompt_data = json.loads(line)
            if last_id is None or prompt_data["id"] > last_id:  # Skip prompts with id <= last_id
                prompts_to_process.append(prompt_data)

    # Process prompts in chunks
    for i in range(0, len(prompts_to_process), chunk_size):
        chunk = prompts_to_process[i:i + chunk_size]

        # Check if we need to rotate to a new file before processing the chunk
        if os.path.exists(current_output_file) and get_file_size_mb(current_output_file) >= max_file_size_mb:
            current_file_num += 1
            current_output_file = f"{base_output_file}{current_file_num:03d}.jsonl"

        # Get completions for all prompts in the chunk
        completions = get_prompts_parallel(
            [item["prompt"] for item in chunk],
            max_workers=3,
            batch_size=10
        )

        # Write all results from this chunk to the same file
        for prompt_data, completion in zip(chunk, completions):
            result = {
                "id": prompt_data["id"],
                "prompt": prompt_data["prompt"],
                "completion": completion
            }
            with open(current_output_file, "a") as f:
                f.write(json.dumps(result) + "\n")

if __name__ == "__main__":
    process_prompts()