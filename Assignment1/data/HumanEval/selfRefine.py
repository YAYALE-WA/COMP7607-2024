from typing import Iterable, Dict
from typing import List, Dict
from openai import *
from openai import OpenAI
import time
from collections import deque
from baseline import write_jsonl, read_problems

HUMAN_EVAL = "HumanEval.jsonl"

def gen_code_with_prompt(client, prompt: str, temperature: int = 0.6) -> Dict[str, str]:
    def build_messages(prompt) -> List[Dict[str, str]]:
        return [{"role": "system", "content": "Environment: ipython"},
                {"role": "user", "content": prompt}]
    
    completion = client.chat.completions.create(
        model="Meta-Llama-3.1-8B-Instruct",
        messages=build_messages(prompt),
        temperature=temperature,
        n=1,
        stream=True
    )
    code = ""
    for chunk in completion:
        delta = chunk.choices[0].delta
        if hasattr(delta, 'content'):
            code += delta.content
    return {"code": code, "prompt": prompt}

def refine_code_with_feedback(client, initial_prompt: str, max_iterations: int = 3, temperature: int = 0) -> Dict[str, str]:
    prompt = initial_prompt
    total_tokens = 0  # 用于统计生成的 token 总数
    start_time = time.time()  # 开始计时

    for iteration in range(max_iterations):
        result = gen_code_with_prompt(client, prompt, temperature)
        code = result["code"]
        total_tokens += len(result["code"].split())

        if iteration < max_iterations - 1:
            feedback = request_feedback_from_model(client, initial_prompt,code)
            total_tokens += len(feedback.split())
            prompt = modify_prompt_with_feedback(code, initial_prompt, feedback)
            print(f"Iteration {iteration + 1}: Refining code with new feedback")
    
    end_time = time.time()
    wall_clock_time = end_time - start_time
    print("Maximum iterations reached, returning latest attempt.")
    return result,{ "wall_clock_time": wall_clock_time,"total_tokens": total_tokens}

def request_feedback_from_model(client,prompt, code: str) -> str:
    feedback_prompt = (
        f"Prompt:\n{prompt}\n\n"
        f"Code:\n{code}\n\n"
        f"Provide feedback; do not suggest or generate any code improvements."
    )
    messages = [
        {"role": "system", "content": "Environment: ipython"},
        {"role": "user", "content": feedback_prompt}
    ]
    
    completion = client.chat.completions.create(
        model="Meta-Llama-3.1-8B-Instruct",
        messages=messages,
        temperature=0,
        n=1,
        stream=False
    )
    
    feedback = completion.choices[0].message.content
    return feedback

def modify_prompt_with_feedback(code: str, Initial_prompt:str, feedback: str) -> str:
    modified_prompt = ( f"Initial Prompt:\n{Initial_prompt}\n\n"
                        f"Code:\n{code}\n\n"
                        f"Please refine the code based on initial prompt and feedback: {feedback}")
    return modified_prompt


def get_self_refine_file():
    problems = read_problems(evalset_file=HUMAN_EVAL)

    client = OpenAI(base_url="https://api.sambanova.ai/v1", api_key="d791f60f-7e79-4ec3-8cda-c5cddd36aa00")
    # Prepare to collect samples
    samples = []
    inference_cost=[]
    # Initialize a deque to keep track of API call timestamps
    call_times = deque()

    def can_make_call():
        current_time = time.time()
        # Remove timestamps older than 60 seconds (1 minute)
        while call_times and call_times[0] <= current_time - 60:
            call_times.popleft()
        # Check if we have made less than 30 calls in the last minute
        return len(call_times) < 6

    def wait_until_can_make_call():
        while not can_make_call():
            # Wait for 3 second before checking again
            time.sleep(3)
        # Record the current timestamp
        call_times.append(time.time())

    # Process each problem with rate limiting
    for task_id in problems:
        prompt = problems[task_id]["prompt"]
        
        sum_wall_clock_time=0
        sum_total_tokens=0

        wait_until_can_make_call()
        try:
            completion,cost = refine_code_with_feedback(client, prompt,max_iterations=3,temperature=0.6)
            # samples.append({"task_id": task_id, "completion": completion["code"]})
            samples.append({"input": prompt, "prompt": completion["prompt"],"output": completion["code"]})
            inference_cost.append({"task_id": task_id, "wall_clock_time": cost["wall_clock_time"],"total_tokens": cost["total_tokens"]})
            sum_wall_clock_time+=cost["wall_clock_time"]
            sum_total_tokens+=cost["total_tokens"]
            print(f"Processed task {task_id}")
        except RateLimitError as e:
            print(f"Rate limit exceeded at task {task_id}: {e}")
            # Wait for a minute before retrying
            time.sleep(60)
            # Clear call times to reset the rate limiter
            call_times.clear()
            # Retry the same prompt
            try:
                completion,cost = refine_code_with_feedback(client, prompt,max_iterations=3,temperature=0.6)
                # samples.append({"task_id": task_id, "completion": completion["code"]})
                samples.append({"input": prompt, "prompt": completion["prompt"],"output": completion["code"]})
                inference_cost.append({"task_id": task_id, "wall_clock_time": cost["wall_clock_time"],"total_tokens": cost["total_tokens"]})
                sum_wall_clock_time+=cost["wall_clock_time"]
                sum_total_tokens+=cost["total_tokens"]
                print(f"Processed task {task_id} after waiting")
            except Exception as e:
                print(f"Error at task {task_id} after retrying: {e}")
        except Exception as e:
            print(f"Error at task {task_id}: {e}")

    wall_cock_time_per_problem=sum_wall_clock_time/164
    total_tokens_per_problem=sum_total_tokens/164
    inference_cost.append({"wall_cock_time_per_problem": wall_cock_time_per_problem, "total_tokens_per_problem": total_tokens_per_problem})
    write_jsonl("answer_selfRefine.jsonl",samples)
    write_jsonl("cost_selfRefine.jsonl",inference_cost)

# get_self_refine_file()
