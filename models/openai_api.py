import os
import openai
import math
import sys
import time
from tqdm import tqdm
from typing import Iterable, List, TypeVar
from tenacity import retry, stop_after_attempt, wait_random_exponential
import func_timeout

T = TypeVar('T')
KEY_INDEX = 0
KEY_POOL = [
    ""  # Input api key here
]
BASE_POOL = [
    ""  # Input api base here
]
PROXY_POOL = [
    {'http': "http://localhost:7890", 'https': "http://localhost:7890"}
]
openai.api_key = KEY_POOL[KEY_INDEX]
openai.api_base = BASE_POOL[KEY_INDEX]
openai.proxy = PROXY_POOL[KEY_INDEX]


# This code is adapted from https://github.com/OSU-NLP-Group/LLM-Knowledge-Conflict/blob/main/code/openai_request.py
class TimeoutError(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutError("The function took too long to run")


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def limited_execution_time(func, model, prompt, temp, max_tokens=128, default=None, **kwargs):
    try:
        if 'gpt-3.5-turbo' in model or 'gpt-4' in model:
            result = func(
                model=model,
                messages=prompt,
                max_tokens=max_tokens,
                temperature=temp
            )
        else:
            result = func(model=model, prompt=prompt, max_tokens=max_tokens, **kwargs)
    except func_timeout.exceptions.FunctionTimedOut:
        return None
    # raise any other exception
    except Exception as e:
        raise e
    return result


def batchify(data: Iterable[T], batch_size: int) -> Iterable[List[T]]:
    # function copied from allenai/real-toxicity-prompts
    assert batch_size > 0
    batch = []
    for item in data:
        # Yield next batch
        if len(batch) == batch_size:
            yield batch
            batch = []
        batch.append(item)

    # Yield last un-filled batch
    if len(batch) != 0:
        yield batch


def openai_unit_price(model_name, token_type="prompt"):
    if model_name in ["gpt-4", "gpt-4-32k"]:
        unit = 0.03 if token_type == "prompt" else 0.06
    elif model_name in ["gpt-4-turbo", "gpt-4-turbo-2024-04-09"]:
        unit = 0.01 if token_type == "prompt" else 0.03
    elif model_name in ["gpt-4o", "gpt-4o-2024-05-13"]:
        unit = 0.005 if token_type == "prompt" else 0.015
    elif model_name == "gpt-3.5-turbo-0125":
        unit = 0.0005 if token_type == "prompt" else 0.0015
    elif model_name == "gpt-3.5-turbo-instruct":
        unit = 0.0015 if token_type == "prompt" else 0.002
    elif 'davinci' in model_name:
        unit = 0.012
    elif 'babbage' in model_name:
        unit = 0.0016
    else:
        unit = -1
    return unit


def calc_cost_w_tokens(total_tokens: int, model_name: str):
    unit = openai_unit_price(model_name, token_type="completion")
    return round(unit * total_tokens / 1000, 4)


def calc_cost_w_prompt(total_tokens: int, model_name: str):
    # 750 words == 1000 tokens
    unit = openai_unit_price(model_name)
    return round(unit * total_tokens / 1000, 4)


def get_perplexity(logprobs):
    assert len(logprobs) > 0, logprobs
    return math.exp(-sum(logprobs) / len(logprobs))


def keep_logprobs_before_eos(tokens, logprobs):
    keep_tokens = []
    keep_logprobs = []
    start_flag = False
    for tok, lp in zip(tokens, logprobs):
        if start_flag:
            if tok == "<|endoftext|>":
                break
            else:
                keep_tokens.append(tok)
                keep_logprobs.append(lp)
        else:
            if tok != '\n':
                start_flag = True
                if tok != "<|endoftext>":
                    keep_tokens.append(tok)
                    keep_logprobs.append(lp)

    return keep_tokens, keep_logprobs


def catch_openai_api_error(prompt_input: list):
    global KEY_INDEX
    error = sys.exc_info()[0]
    if error == openai.error.InvalidRequestError:
        # something is wrong: e.g. prompt too long
        print(f"InvalidRequestError\nPrompt:\n\n{prompt_input}\n\n")
        assert False
    elif error == openai.error.RateLimitError:
        KEY_INDEX = (KEY_INDEX + 1) % len(KEY_POOL)
        openai.api_key = KEY_POOL[KEY_INDEX]
        print("RateLimitError", openai.api_key)
    elif error == openai.error.APIError:
        KEY_INDEX = (KEY_INDEX + 1) % len(KEY_POOL)
        openai.api_key = KEY_POOL[KEY_INDEX]
        print("APIError", openai.api_key)
    elif error == openai.error.AuthenticationError:
        KEY_INDEX = (KEY_INDEX + 1) % len(KEY_POOL)
        openai.api_key = KEY_POOL[KEY_INDEX]
        print("AuthenticationError", openai.api_key)
    elif error == TimeoutError:
        KEY_INDEX = (KEY_INDEX + 1) % len(KEY_POOL)
        openai.api_key = KEY_POOL[KEY_INDEX]
        print("TimeoutError, retrying...")
    else:
        print("API error:", error)


def prompt_chatgpt(user_input, temperature, max_tokens, history=[], system_input="", model_name='gpt-3.5-turbo-0125'):
    '''
    :param user_input: you texts here
    :param history: ends with assistant output.
                    e.g. [{"role": "system", "content": xxx},
                          {"role": "user": "content": xxx},
                          {"role": "assistant", "content": "xxx"}]
    :param system_input: "You are a helpful assistant."
    return: assistant_output, (updated) history, money cost
    '''
    
    if len(history) == 0:
        if len(system_input) != 0:
            history = [{"role": "system", "content": system_input}]
            history.append({"role": "user", "content": user_input})
        else:
            history = [{"role": "user", "content": user_input}]
    else:
        history.append({"role": "user", "content": user_input})
            
    while True:
        try:
            completion = limited_execution_time(openai.ChatCompletion.create,
                                                model=model_name,
                                                prompt=history,
                                                temp=temperature,
                                                max_tokens=max_tokens)
            if completion is None:
                raise TimeoutError
            break
        except:
            catch_openai_api_error(user_input)
            time.sleep(1)

    assistant_output = completion['choices'][0]['message']['content']
    history.append({"role": "assistant", "content": assistant_output})
    total_prompt_tokens = completion['usage']['prompt_tokens']
    total_completion_tokens = completion['usage']['completion_tokens']

    # with open(save_path, 'a+', encoding='utf-8') as f:
    #     assistant_output = str(index) + "\t" + "\t".join(x for x in assistant_output.split("\n"))
    #     f.write(assistant_output + '\n')

    return assistant_output, history, calc_cost_w_tokens(total_prompt_tokens, model_name) + calc_cost_w_prompt(total_completion_tokens, model_name)
