import os

from bespokelabs import curator
from pydantic import BaseModel


class JudgeResult(BaseModel):
    """Result of the judge's evaluation."""

    correct: bool
    reasoning: str


class PuzzleJudge(curator.LLM):
    """Curator class for processing RiddleSense dataset."""

    response_format = JudgeResult

    def prompt(self, input):
        """Create a prompt for the judge to evaluate the correctness of a solution."""
        return f"""
        You are a judge that evaluates the correctness of a solution.
        You will be given a proposed solution and a ground truth solution.
        You will need to determine if the proposed solution is correct.
        The proposed solution must arrive at the ground truth solution.

        PROPOSED SOLUTION: {input["deepseek_solution"]}
        GROUND TRUTH SOLUTION: {input["answer"]}
        """

    def parse(self, input, response):
        """Parse the judge's response to extract correctness and reasoning."""
        return {
            **input,
            "correct": response.correct,
            "judge_reasoning": response.reasoning,
        }


def mocked_judge(ds):
    ds = ds.add_column("correct", [True] * len(ds))
    ds = ds.add_column("judge_reasoning", ["Hmmmmm."] * len(ds))
    return ds


if os.environ.get("MOCK_VERIFY"):
    puzzle_judge = mocked_judge
else:
    puzzle_judge = PuzzleJudge(model_name="gpt-4o")
