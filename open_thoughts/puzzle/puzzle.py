import os

from datasets import load_dataset

from open_thoughts import decontaminate, deduplicate
from open_thoughts.puzzle.reason import reason


def riddle_sense_map(x):
    question = x["question"]
    choices = x["choices"]
    full_question = question
    labels = choices["label"]
    texts = choices["text"]
    for label, text in zip(labels, texts):
        full_question += f"\n{label}: {text}"
    return {"question": full_question, "answer": x["answerKey"]}


if __name__ == "__main__":
    ds = load_dataset("INK-USC/riddle_sense", split="train", trust_remote_code=True)
    ds = ds.map(riddle_sense_map)
    ds = ds.shuffle(seed=42).take(1_250)
    ds = ds.remove_columns(["answerKey", "choices"])
    ds = ds.add_column("domain", ["puzzle"] * len(ds))
    ds = ds.add_column("source", ["riddle_sense"] * len(ds))
    ds = deduplicate(ds)
    ds = decontaminate(ds)
    ds = reason(ds)
    ds.push_to_hub(f"{os.environ.get('HF_ORG')}/open-thoughts-puzzle", private=os.environ.get("HF_PRIVATE"))
