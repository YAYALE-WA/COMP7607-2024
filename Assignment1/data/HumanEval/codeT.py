from typing import Iterable, Dict
from typing import List, Dict
from openai import *
from openai import OpenAI
import time
from collections import deque
from baseline import write_jsonl, read_problems


HUMAN_EVAL = "HumanEval.jsonl"

import signal
import contextlib

class TimeoutException(Exception):
    pass

@contextlib.contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")

    # 设置超时信号处理器
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(int(seconds))
    try:
        yield
    finally:
        signal.alarm(0)  # 取消闹钟

def gen_code_with_promt(client,
                             prompt: str,
                             temperature: float = 0.8, 
                             n: int = 5) -> Dict[str, str]:

    def build_messages(prompt) -> List[Dict[str, str]]:
        messages = [
            {"role": "system", "content": "Environment: ipython"},
            {"role": "user", "content": prompt}
        ]
        return messages
    
    token=0
    solutions = []
    for _ in range(n):
        completion = client.chat.completions.create(
            model="Meta-Llama-3.1-8B-Instruct",
            messages=build_messages(prompt),
            temperature=temperature, 
            stream=False
        )

        token+=len(completion.choices[0].message.content.split())
        solutions.append(completion.choices[0].message.content)

    return solutions,token

def process_test_cases(output):
    """
    Processes the output to remove lines before '# Test cases' 
    and ensures no 'not' in the assert statements.
    """
    test_cases_start = False
    processed_output = []
    test_cases=[]
    for line in output.splitlines():
        if "assert" not in line and "==" not in line:
                continue
        else:
            processed_output.append(line)
            test_cases.append(line)

    return "\n".join(processed_output),test_cases

def gen_test_cases(client,
                    prompt: str,
                    entry_point):
    test_prompt = (
        prompt + "    pass\n# Do not implement code.\n# check the corrcetness of " + entry_point + "\nRequirements for the test cases:" +
        "\n1. Each test case should be written as `assert function_call == expected_result`, where `function_call` is the function invocation, and `expected_result` is the correct output of that function call." +
        "\n2. Cover a variety of input scenarios, including:" +
        "\n   - Typical cases that reflect normal inputs." +
        "\n   - Edge cases such as empty inputs, very large or small values, and boundary values." +
        "\n   - Any specific edge cases mentioned in the function description." +
        "\n3. Ensure that the test cases are accurate, and the expected result matches the function's logic. Avoid incorrect test cases." +
        "\n4. Format each test case as a single `assert` statement without additional comments or explanations." +
        "\n\nOutput only the `assert` statements, one per line, and ensure each statement uses `==` for comparison."
    )
    token=0
    messages = [
        {"role": "system", "content": "Environment: ipython"},
        {"role": "user", "content": test_prompt}
    ]
    
    completion = client.chat.completions.create(
        model="Meta-Llama-3.1-8B-Instruct",
        messages=messages,
        stream=True
    )
    test_cases = ""
    for chunk in completion:
        delta = chunk.choices[0].delta
        if hasattr(delta, 'content'):
            test_cases += delta.content
    
    token+=len(test_cases.split())

    _,test_cases=process_test_cases(test_cases)

    return test_cases,token,test_prompt


def check(code,test_cases,entry_point):
    passed_tests = 0
    exec_globals = {}

    import io
    import contextlib
    # 创建空的字符串流来屏蔽输出

    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        for test_case in test_cases:
            try:
                with time_limit(5):  
                    check_program = (
                        code + "\n" +
                        "def check("+entry_point+"):\n"+
                        "    "+test_case + "\n" +
                        f"check({entry_point})"
                    )
                    exec(check_program, exec_globals)
                    passed_tests += 1
            except TimeoutException:  # 捕获超时错误
                print("Timeout")
                continue  
            except Exception as e:
                print(e)
                continue  

    score=passed_tests/len(test_cases)
    return score


def get_best_code(client, prompt, entry_point,problem):
    total_tokens = 0  # 用于统计生成的 token 总数
    start_time = time.time()  # 开始计时
    
    solutions,token_code=gen_code_with_promt(client, prompt)
    test_cases,token_test_cases,test_prompt=gen_test_cases(client, prompt, entry_point)
    
    total_tokens+=token_code+token_test_cases

    best_code=solutions[0] 

    max_score=0
    for solution in solutions:
        score=check(solution,test_cases,entry_point)

        if score>max_score: 
            max_score=score
            best_code=solution
        print(score)

    end_time = time.time()
    wall_clock_time = end_time - start_time

    best_completion={"code": best_code, "prompt": test_prompt}
    return best_completion,{ "wall_clock_time": wall_clock_time,"total_tokens": total_tokens}


def get_codeT_file():
    problems = read_problems(evalset_file=HUMAN_EVAL)

    client = OpenAI(base_url="https://api.sambanova.ai/v1", api_key="d791f60f-7e79-4ec3-8cda-c5cddd36aa00")
    # Prepare to collect samples
    samples = []
    inference_cost=[]
    sum_wall_clock_time=0
    sum_total_tokens=0

    # Initialize a deque to keep track of API call timestamps
    call_times = deque()

    def can_make_call():
        current_time = time.time()
        # Remove timestamps older than 60 seconds (1 minute)
        while call_times and call_times[0] <= current_time - 60:
            call_times.popleft()
        # Check if we have made less than 30 calls in the last minute
        return len(call_times) < 5

    def wait_until_can_make_call():
        while not can_make_call():
            # Wait for 3 second before checking again
            time.sleep(3)
        # Record the current timestamp
        call_times.append(time.time())

    cnt=0
    # Process each problem with rate limiting
    for task_id in problems:
        prompt = problems[task_id]["prompt"]
        entry_point = problems[task_id]["entry_point"]
        wait_until_can_make_call()
        try:
            completion,cost = get_best_code(client, prompt, entry_point,problems[task_id])
            # samples.append({"task_id": task_id, "completion": code})
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
                completion,cost = get_best_code(client, prompt, entry_point,problems[task_id])
                # samples.append({"task_id": task_id, "completion": code})
                samples.append({"input": prompt, "prompt": completion["prompt"],"output": completion["code"]})
                inference_cost.append({"task_id": task_id, "wall_clock_time": cost["wall_clock_time"],"total_tokens": cost["total_tokens"]})
                sum_wall_clock_time+=cost["wall_clock_time"]
                sum_total_tokens+=cost["total_tokens"]
                print(f"Processed task {task_id} after waiting")
            except Exception as e:
                print(f"Error at task {task_id} after retrying: {e}")
        except Exception as e:
            print(f"Error at task {task_id}: {e}")

        cnt+=1
        if cnt==4:
            write_jsonl("answer-codeT.jsonl", samples,True)
            write_jsonl("cost_codeT.jsonl",inference_cost,True)
            samples=[]
            inference_cost=[]
        elif cnt%10==0:
            write_jsonl("answer-codeT.jsonl", samples,True)
            write_jsonl("cost_codeT.jsonl",inference_cost,True)
            samples=[]
            inference_cost=[]
        else:
            continue

    wall_cock_time_per_problem=sum_wall_clock_time/164
    total_tokens_per_problem=sum_total_tokens/164
    inference_cost.append({"wall_cock_time_per_problem": wall_cock_time_per_problem, "total_tokens_per_problem": total_tokens_per_problem})
    write_jsonl("cost_codeT.jsonl",inference_cost,True)
    # Write samples to file
    write_jsonl("answer-codeT.jsonl", samples,True)
