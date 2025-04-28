import time

import openai
from openai.types.chat import ChatCompletion


def make_request(
    client: openai.Client,
    message: str,
    model: str,
    max_tokens: int = 512,
    temperature: float = 1,
    reasoning_effort: str = "medium",
    n: int = 1,
    **kwargs
) -> ChatCompletion:
    kwargs["top_p"] = 0.95
    kwargs["max_completion_tokens"] = max_tokens
    kwargs["temperature"] = temperature
    if any(model.startswith(m) or model.endswith(m) for m in ["o1-", "o3-", "reasoner", "grok-3-mini-beta"]):  # pop top-p and max_completion_tokens
        kwargs.pop("top_p")
        kwargs.pop("max_completion_tokens")
        kwargs.pop("temperature")
        kwargs["reasoning_effort"] = reasoning_effort
    
    system_prompt = '''\
A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. User: Please integrate natural language reasoning with programs to solve the problem above. For math problems, please put your final answer within \\boxed{}. For code problems, please put your final answer in a markdown code block like this: ```python\nyour code here\n```."'''

    
    return client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": message},
        ],
        n=n,
        **kwargs
    )


def make_auto_request(*args, **kwargs) -> ChatCompletion:
    ret = None
    while ret is None:
        try:
            ret = make_request(*args, **kwargs)
        except openai.RateLimitError:
            print("Rate limit exceeded. Waiting...")
            time.sleep(5)
        except openai.APIConnectionError:
            print("API connection error. Waiting...")
            time.sleep(5)
        except openai.APIError as e:
            print(e)
        except Exception as e:
            print("Unknown error. Waiting...")
            print(e)
            time.sleep(1)
    return ret