import os

from datasets import concatenate_datasets, load_dataset

from open_thoughts import prompt, verify

org = os.environ.get("HF_ORG")
unverified_mix_ds = []
verified_mix_ds = []
for subset in ["puzzle", "science", "math", "code"]:
    ds = load_dataset(f"{org}/open-thoughts-{subset}", split="train")
    verified_ds = verify(ds)

    ds = ds.map(prompt.map_to_share_gpt)
    ds = ds.select_columns(["system", "conversations"])
    unverified_mix_ds.append(ds)

    verified_ds = verified_ds.map(prompt.map_to_share_gpt)
    if len(verified_ds) > 0:
        verified_ds = verified_ds.select_columns(["system", "conversations"])
        verified_mix_ds.append(verified_ds)

unverified_mix = concatenate_datasets(unverified_mix_ds)
unverified_mix.push_to_hub(f"{org}/open-thoughts-unverified-mix")

verified_mix = concatenate_datasets(verified_mix_ds)
verified_mix.push_to_hub(f"{org}/open-thoughts-verified-mix")
