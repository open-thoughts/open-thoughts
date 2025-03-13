import argparse
import os
import platform
from multiprocessing import freeze_support

from datasets import concatenate_datasets, load_dataset

from open_thoughts import prompt, verify

parser = argparse.ArgumentParser()
parser.add_argument("--dry-run", action="store_true", default=False)
args = parser.parse_args()


if __name__ == "__main__":
    # on a mac, freeze support
    if platform.system() == "Darwin":
        freeze_support()

    org = os.environ.get("HF_ORG")
    unverified_mix_ds = []
    verified_mix_ds = []
    for subset in ["puzzle", "science", "math", "code"]:
        ds = load_dataset(f"{org}/open-thoughts-{subset}{'-dry-run' if args.dry_run else ''}", split="train")
        verified_ds = verify(ds)

        ds = ds.map(prompt.map_to_share_gpt)
        ds = ds.select_columns(["system", "conversations"])
        unverified_mix_ds.append(ds)

        verified_ds = verified_ds.map(prompt.map_to_share_gpt)
        if len(verified_ds) > 0:
            verified_ds = verified_ds.select_columns(["system", "conversations"])
            verified_mix_ds.append(verified_ds)

    unverified_mix = concatenate_datasets(unverified_mix_ds)
    unverified_mix.push_to_hub(f"{org}/open-thoughts-unverified-mix{'-dry-run' if args.dry_run else ''}", private=os.environ.get("HF_PRIVATE"))

    verified_mix = concatenate_datasets(verified_mix_ds)
    verified_mix.push_to_hub(f"{org}/open-thoughts-verified-mix{'-dry-run' if args.dry_run else ''}", private=os.environ.get("HF_PRIVATE"))
