import multiprocessing as mp
from functools import partial
from multiprocessing import Pool

from datasets import Dataset, load_dataset
from tqdm import tqdm

from open_thoughts.deduplicate import fuzz_string_pair
from open_thoughts.eval import EVALUATION_DATASETS


def decontaminate(dataset: Dataset, column="question", evals=EVALUATION_DATASETS, threshold=95.0) -> Dataset:
    """Remove rows from dataset that have similar strings in eval_datasets based on fuzzy matching."""
    n_processes = mp.cpu_count()

    # Get values from input dataset
    dataset_strings = [str(x) for x in dataset[column] if x is not None]
    indices_to_remove = set()

    for eval_name, eval_info in evals.items():
        eval_splits = eval_info["eval_splits"]
        eval_columns = eval_info["eval_columns"]
        eval_subset = eval_info.get("eval_subset", None)
        if eval_subset is not None:
            ds = load_dataset(eval_name, eval_subset, split=eval_splits, trust_remote_code=True)
        else:
            ds = load_dataset(eval_name, split=eval_splits, trust_remote_code=True)

        # for each split, column, and value
        eval_strings = [str(x) for split in ds for column in eval_columns for x in split[column] if x is not None]

        # Track indices to remove
        process_pair = partial(
            fuzz_string_pair,
            values2=eval_strings,
            similarity_threshold=threshold,
        )

        with Pool(n_processes) as pool:
            matches = list(
                tqdm(
                    pool.imap(process_pair, dataset_strings, chunksize=100),
                    total=len(dataset_strings),
                    desc=f"Decontaminating against {eval_name}",
                )
            )

        # Find indices where matches were found
        for i, match_list in enumerate(matches):
            if any(score >= threshold for _, _, score in match_list):
                indices_to_remove.add(i)

    keep_mask = [i for i in range(len(dataset)) if i not in indices_to_remove]
    clean_dataset = dataset.select(keep_mask)

    print(f"Removed {len(indices_to_remove)} contaminated rows")
    print(f"Original size: {len(dataset)}, New size: {len(clean_dataset)}")

    return clean_dataset
