import argparse
import os

from datasets import concatenate_datasets

from open_thoughts import decontaminate, deduplicate
from open_thoughts.science.camel import load, subsample
from open_thoughts.science.reason import reason

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", default=False)
    args = parser.parse_args()

    domains = {
        "biology": None,
        "chemistry": None,
        "physics": None,
    }
    for domain in domains:
        ds = load(domain)
        ds = subsample(ds, 2)
        ds = ds.add_column("domain", [domain] * len(ds))
        domains[domain] = ds
    ds = concatenate_datasets(domains.values())

    if args.dry_run:
        ds = ds.take(2)

    ds = ds.rename_column("message_1", "question")
    ds = ds.rename_column("topic;", "topic")
    ds = ds.select_columns(["question", "domain", "topic", "sub_topic"])
    ds = ds.add_column("source", ["camel"] * len(ds))

    ds = deduplicate(ds)
    ds = decontaminate(ds)
    ds = reason(ds)

    if args.dry_run:
        print("======== SCIENCE DATASET ========")
        print(ds)
        print(ds[0])
        print("================")

    ds.push_to_hub(f"{os.environ.get('HF_ORG')}/open-thoughts-science{'-dry-run' if args.dry_run else ''}", private=os.environ.get("HF_PRIVATE"))
