from typing import Iterable, Dict
from typing import List, Dict
from openai import *
from openai import OpenAI
import time
from collections import deque
from baseline import write_jsonl, read_problems
from execution import time_limit

HUMAN_EVAL = "HumanEval_new.jsonl"

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
    solutions=[]
    for iteration in range(max_iterations):
        result = gen_code_with_prompt(client, prompt, temperature)
        code = result["code"]
        solutions.append(code)
        if iteration < max_iterations - 1:
            feedback = request_feedback_from_model(client, initial_prompt,code)
            prompt = modify_prompt_with_feedback(code, initial_prompt, feedback)
            print(f"Iteration {iteration + 1}: Refining code with new feedback")

    print("Maximum iterations reached, returning latest attempt.")
    return solutions

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

def process_test_cases(output):
    """
    Processes the output to remove lines before '# Test cases' 
    and ensures no 'not' in the assert statements.
    """
    test_cases_start = False
    processed_output = []
    test_cases=[]
    for line in output.splitlines():
        if "assert" not in line:
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

    _,test_cases=process_test_cases(test_cases)

    return test_cases


def check(code,test_cases,entry_point):
    passed_tests = 0
    exec_globals = {}

    import io
    import contextlib
    # 创建空的字符串流来屏蔽输出

    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        for test_case in test_cases:
            try:
                with time_limit(5):  # 假设 time_limit 是一个控制时间的上下文管理器
                    check_program = (
                        code + "\n" +
                        "def check("+entry_point+"):\n"+
                        "    "+test_case + "\n" +
                        f"check({entry_point})"
                    )
                    exec(check_program, exec_globals)
                    passed_tests += 1
            except Exception as e:
                print(f"Test case {test_case} failed with exception: {e}")
                continue  # 跳过失败的测试案例

    score=passed_tests/len(test_cases)
    return score


def get_best_code(client, prompt, entry_point):
    solutions=refine_code_with_feedback(client, prompt,max_iterations=5,temperature=0.6)
    test_cases=gen_test_cases(client, prompt, entry_point)
    best_code=solutions[0] 
    
    max_score=0
    for solution in solutions:
        score=check(solution,test_cases,entry_point)
        print(score)
        if score>max_score: 
            max_score=score
            best_code=solution
         
    return best_code


def get_combine_file():
    problems = read_problems(evalset_file=HUMAN_EVAL)

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
        return len(call_times) < 3

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
            completion = get_best_code(client, prompt, entry_point)
            samples.append({"task_id": task_id, "completion": completion})
            print(f"Processed task {task_id}")
        except RateLimitError as e:
            print(f"Rate limit exceeded at task {task_id}: {e}")
            # Wait for a minute before retrying
            time.sleep(60)
            # Clear call times to reset the rate limiter
            call_times.clear()
            # Retry the same prompt
            try:
                completion = get_best_code(client, prompt, entry_point)
                samples.append({"task_id": task_id, "completion": completion})
                print(f"Processed task {task_id} after waiting")
            except Exception as e:
                print(f"Error at task {task_id} after retrying: {e}")
        except Exception as e:
            print(f"Error at task {task_id}: {e}")
        
        cnt+=1
        if cnt==10:
            write_jsonl("test-combine.jsonl",samples,True)
            samples=[]
        elif cnt%10==0:
            write_jsonl("test-combine.jsonl",samples,True)


    write_jsonl("test-combine.jsonl",samples,True)

