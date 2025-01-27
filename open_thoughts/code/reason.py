import os

from bespokelabs import curator

from open_thoughts.prompt import DEEPSEEK_R1_SYSTEM_PROMPT, format_code_prompt
from open_thoughts.reason import mocked_reasoner


class Reasoner(curator.LLM):
    return_completions_object = True

    def prompt(self, input):
        """Create a prompt for the LLM to reason about the problem."""
        formatted_prompt = format_code_prompt(input)
        return [
            {"role": "system", "content": DEEPSEEK_R1_SYSTEM_PROMPT},
            {"role": "user", "content": formatted_prompt},
        ]

    def parse(self, input, response):
        """Parse the LLM response to extract reasoning and solution."""
        input["reasoning"] = response["choices"][0]["message"]["reasoning_content"]
        input["deepseek_solution"] = response["choices"][0]["message"]["content"]
        input["formatted_prompt"] = format_code_prompt(input)
        return input


def reason(ds):
    if os.environ.get("MOCK_REASON"):
        return mocked_reasoner(ds)
    reasoner = Reasoner(
        model_name="deepseek-reasoner",
        backend_params={
            "max_requests_per_minute": 600,
            "max_tokens_per_minute": 10000000,
            "request_timeout": 30 * 60,
        },
        generation_params={
            "temp": 0.0,
            "max_tokens": 8192,
        },
    )
    ds = reasoner(ds)
    return ds
