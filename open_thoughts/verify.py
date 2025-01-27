from datasets import Dataset

from open_thoughts.code.judge import code_judge
from open_thoughts.math.judge import math_judge
from open_thoughts.puzzle.judge import puzzle_judge


def verify(ds: Dataset):
    if ds["domain"][0] == "math":
        ds = math_judge(ds)
    elif ds["domain"][0] == "puzzle":
        ds = puzzle_judge(ds)
    elif ds["domain"][0] == "code":
        ds = code_judge(ds)
    else:
        ds = ds.add_column("correct", [True] * len(ds))

    ds = ds.filter(lambda x: x["correct"])
    return ds
