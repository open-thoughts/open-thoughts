import multiprocessing as mp
from collections import defaultdict
from functools import partial
from multiprocessing import Pool
from typing import List, Tuple

from datasets import Dataset
from rapidfuzz import fuzz, process
from tqdm import tqdm


def fuzz_string_pair(str1: str, values2: List[str], similarity_threshold: float) -> List[Tuple]:
    matches_with_scores = process.extract(str1, values2, scorer=fuzz.ratio, score_cutoff=similarity_threshold)
    return [(str1, match_tuple[0], match_tuple[1]) for match_tuple in matches_with_scores]


def deduplicate(dataset: Dataset, column="question", similarity_threshold: float = 95.0) -> Dataset:
    """Fuzzy deduplicate dataset rows based on fuzzy string matching within specified column."""
    values = [str(x) for x in dataset[column] if x is not None]
    unique_values = list(set(values))
    n_processes = mp.cpu_count()

    process_pair = partial(
        fuzz_string_pair,
        values2=unique_values,
        similarity_threshold=similarity_threshold,
    )
    with Pool(n_processes) as pool:
        all_matches = list(
            tqdm(
                pool.imap(process_pair, unique_values, chunksize=100),
                total=len(unique_values),
                desc="Finding duplicates",
            )
        )

    str_to_indices = defaultdict(list)
    for i, val in enumerate(values):
        str_to_indices[val].append(i)

    indices_to_remove = set()
    for matches_list in all_matches:
        for str1, str2, score in matches_list:
            if score >= similarity_threshold:
                indices1 = str_to_indices[str1]
                indices2 = str_to_indices[str2]
                all_indices = list(set(indices1 + indices2))
                all_indices.sort()
                indices_to_remove.update(all_indices[1:])
    keep_mask = [i for i in range(len(dataset)) if i not in indices_to_remove]
    clean_dataset = dataset.select(keep_mask)

    print(f"Removed {len(indices_to_remove)} duplicate rows")
    print(f"Original size: {len(dataset)}, New size: {len(clean_dataset)}")
    return clean_dataset
