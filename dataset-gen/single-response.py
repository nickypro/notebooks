# %%
# Assume openai>=1.0.0
from openai import OpenAI
from dotenv import load_dotenv

# Load token from env file
load_dotenv("./env")
deepinfra_token = os.environ.get("DEEPINFRA_TOKEN")

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

def get_llama_completion(prompt: str) -> str:
    # Create an OpenAI client with your deepinfra token and endpoint
    openai = OpenAI(
        api_key=deepinfra_token,
        base_url="https://api.deepinfra.com/v1/openai",
    )

    chat_completion = openai.chat.completions.create(
        model="meta-llama/Llama-3.2-3B-Instruct",
        messages=[{"role": "user", "content": prompt}],
    )

    return chat_completion.choices[0].message.content

# %%
get_llama_completion("How many weeks in a fiscal year? Is it always the same?")
