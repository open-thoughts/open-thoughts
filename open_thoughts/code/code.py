import argparse
import os

from open_thoughts.code.combine import combine
from open_thoughts.code.reason import reason
from open_thoughts.code.standardize import standardize
from open_thoughts.decontaminate import decontaminate
from open_thoughts.deduplicate import deduplicate
from open_thoughts.prompt import SKY_T1_SYSTEM_PROMPT, format_code_prompt


def map_code_to_share_gpt(row):
    user_message = format_code_prompt(row)
    assistant_message = (
        f"<|begin_of_thought|>\n\n{row['reasoning']}\n\n<|end_of_thought|>\n\n<|begin_of_solution|>\n\n{row['deepseek_solution']}\n\n<|end_of_solution|>"
    )

    return {
        "system": SKY_T1_SYSTEM_PROMPT,
        "conversations": [
            {"from": "user", "value": user_message},
            {"from": "assistant", "value": assistant_message},
        ],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", default=False)
    args = parser.parse_args()

    # Each of the subsets below is formatted in a different way, we first standardize them
    # into a common format and combine them.
    # dry run only on apps

    if args.dry_run:
        subsets = {
            "MatrixStudio/Codeforces-Python-Submissions": None,
            "BAAI/TACO": None,
            "codeparrot/apps": None,
        }
    else:
        subsets = {
            "MatrixStudio/Codeforces-Python-Submissions": None,
            "BAAI/TACO": None,
            "codeparrot/apps": None,
            "deepmind/code_contests": None,
        }

    for subset in subsets:
        print(f"Standardizing {subset}...")
        ds = standardize(subset, num_hf_proc_workers=os.cpu_count(), dry_run=args.dry_run)
        ds = ds.add_column("subset", [subset] * len(ds))

        if args.dry_run:
            subsets[subset] = ds.take(3)
        else:
            subsets[subset] = ds

    ds = combine(subsets, dry_run=args.dry_run)
    # Deduplicate and decontaminate the dataset against benchmarks.
    ds = deduplicate(ds, column="problem")
    ds = decontaminate(ds, column="problem")

    # Annotate the dataset with reasoning.
    ds = reason(ds)

    if args.dry_run:
        print("======== CODE DATASET ========")
        print(ds)
        print(ds[0])
        print("================")

    ds.push_to_hub(f"{os.environ.get('HF_ORG')}/open-thoughts-code-annotations{'-dry-run' if args.dry_run else ''}", private=os.environ.get("HF_PRIVATE"))
    ds = ds.add_column("domain", ["code"] * len(ds))
    ds.push_to_hub(f"{os.environ.get('HF_ORG')}/open-thoughts-code{'-dry-run' if args.dry_run else ''}", private=os.environ.get("HF_PRIVATE"))
