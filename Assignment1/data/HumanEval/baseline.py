import pdb
from typing import Iterable, Dict
import gzip
import json
import os
from typing import List, Dict
from openai import *
from openai import OpenAI
import time
from collections import deque

HUMAN_EVAL = "HumanEval.jsonl"

def write_jsonl(filename: str, data: Iterable[Dict], append: bool = False):
    """
    Writes an iterable of dictionaries to jsonl
    """
    if append:
        mode = 'ab'
    else:
        mode = 'wb'
    filename = os.path.expanduser(filename)
    if filename.endswith(".gz"):
        with open(filename, mode) as fp:
            with gzip.GzipFile(fileobj=fp, mode='wb') as gzfp:
                for x in data:
                    gzfp.write((json.dumps(x) + "\n").encode('utf-8'))
    else:
        with open(filename, mode) as fp:
            for x in data:
                fp.write((json.dumps(x) + "\n").encode('utf-8'))

def stream_jsonl(filename: str) -> Iterable[Dict]:
    """
    Parses each jsonl line and yields it as a dictionary
    """
    if filename.endswith(".gz"):
        with open(filename, "rb") as gzfp:
            with gzip.open(gzfp, 'rt') as fp:
                for line in fp:
                    if any(not x.isspace() for x in line):
                        yield json.loads(line)
    else:
        with open(filename, "r") as fp:
            for line in fp:
                if any(not x.isspace() for x in line):
                    yield json.loads(line)



def read_problems(evalset_file: str = HUMAN_EVAL) -> Dict[str, Dict]:
    return {task["task_id"]: task for task in stream_jsonl(evalset_file)}


def send_prompt_with_context(client,
                             prompt: str,
                             temperature: int = 0) -> Dict[str, str]:

    def build_messages(prompt) -> List[Dict[str, str]]:
        messages = [
            {"role": "system", "content": "Environment: ipython"},
            {"role": "user", "content": prompt}
        ]
        return messages
    
    completion = client.chat.completions.create(
        model="Meta-Llama-3.1-8B-Instruct",
        messages=build_messages(prompt),
        temperature=temperature,
        n=1,
        stream=True
    )
    full_response = ""
    for chunk in completion:
        delta = chunk.choices[0].delta
        if hasattr(delta, 'content'):
            full_response += delta.content

    return full_response



def get_zeroshot_baseline_file():
    problems = read_problems()

    client = OpenAI(base_url="https://api.sambanova.ai/v1", api_key="d791f60f-7e79-4ec3-8cda-c5cddd36aa00")
    # Prepare to collect samples
    samples = []

    # Initialize a deque to keep track of API call timestamps
    call_times = deque()

    def can_make_call():
        current_time = time.time()
        # Remove timestamps older than 60 seconds (1 minute)
        while call_times and call_times[0] <= current_time - 60:
            call_times.popleft()
        # Check if we have made less than 30 calls in the last minute
        return len(call_times) < 30

    def wait_until_can_make_call():
        while not can_make_call():
            # Wait for 1 second before checking again
            time.sleep(3)
        # Record the current timestamp
        call_times.append(time.time())

    # Process each problem with rate limiting
    for task_id in problems:
        prompt = problems[task_id]["prompt"]
        wait_until_can_make_call()
        try:
            completion = send_prompt_with_context(client, prompt)
            samples.append({"task_id": task_id, "completion": completion})
            print(f"Processed task {task_id}")
        except RateLimitError as e:
            print(f"Rate limit exceeded at task {task_id}: {e}")
            # Wait for a minute before retrying
            time.sleep(20)
            # Clear call times to reset the rate limiter
            call_times.clear()
            # Retry the same prompt
            try:
                completion = send_prompt_with_context(client, prompt)
                samples.append({"task_id": task_id, "completion": completion})
                print(f"Processed task {task_id} after waiting")
            except Exception as e:
                print(f"Error at task {task_id} after retrying: {e}")
        except Exception as e:
            print(f"Error at task {task_id}: {e}")

    # Write samples to file
    write_jsonl("zeroshot.baseline.jsonl", samples)

# get_zeroshot_baseline_file()