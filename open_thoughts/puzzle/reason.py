import os

from bespokelabs import curator

from open_thoughts import prompt
from open_thoughts.reason import mocked_reasoner


class Reasoner(curator.LLM):
    return_completions_object = True

    def prompt(self, input):
        """Create a prompt for the LLM to reason about the problem."""
        breakpoint()
        return [
            {"role": "system", "content": prompt.DEEPSEEK_R1_SYSTEM_PROMPT},
            {"role": "user", "content": input["question"]},
        ]

    def parse(self, input, response):
        """Parse the LLM response to extract reasoning and solution."""
        return {
            "question": input["question"],
            "reasoning": response["choices"][0]["message"]["reasoning_content"],
            "deepseek_solution": response["choices"][0]["message"]["content"],
            "answer": input["answer"],
            "domain": input["domain"],
        }


def reason(ds):
    if os.environ.get("MOCK_REASON"):
        return mocked_reasoner(ds, answer_column="answer")
    reasoner = Reasoner(
        model_name="deepseek-reasoner",
        generation_params={"temp": 0.0, "max_tokens": 8_000},
        backend_params={"max_requests_per_minute": 500, "max_tokens_per_minute": 100_000_000},
    )
    return reasoner(ds)
