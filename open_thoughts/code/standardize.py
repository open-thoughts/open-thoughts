import ast
import hashlib
import json
import urllib
from typing import Dict, List

import datasets

from open_thoughts.code.constants import COLUMNS, code_contests_languages_map, code_contests_sources_map
from open_thoughts.code.filters import filter_num_solutions, filter_problem, filter_solutions, filter_tests


def map_languages(solutions: Dict[str, list]) -> dict:
    solutions_out = {"language": [], "solution": []}
    for language, solution in zip(solutions["language"], solutions["solution"]):
        language = code_contests_languages_map.get(str(language))
        solutions_out["language"].append(language)
        solutions_out["solution"].append(solution)

    return solutions_out


def codecontests_map_sources(source: str) -> str:
    return code_contests_sources_map.get(str(source))


def codecontests_map_languages(language: str) -> str:
    return code_contests_languages_map.get(str(language))


def get_domain(url):
    return urllib.parse.urlparse(url).netloc


def parse_input_output(x):
    try:
        return json.loads(x)
    except:
        return {}


def parse_solutions(x):
    try:
        return len(json.loads(x))
    except:
        return 0


def apps_process_solutions(solutions: str) -> List[str]:
    # print(type(solutions))

    if isinstance(solutions, str):
        try:
            solutions = json.loads(solutions)
        except:
            # print(solutions)
            try:
                solutions = ast.literal_eval(solutions)
            except:
                solutions = []

    return solutions


def compute_problem_id(description: str) -> str:
    return hashlib.md5(description.encode()).hexdigest()


def dump_tests(tests: Dict[str, List[str]]) -> str:
    return json.dumps(tests)


def codecontests_combine_tests(
    public_tests: List[Dict[str, List[str]]],
    private_tests: List[Dict[str, List[str]]],
    generated_tests: List[Dict[str, List[str]]],
) -> Dict[str, List[str]]:
    return {
        "inputs": public_tests["input"] + private_tests["input"] + generated_tests["input"],
        "outputs": public_tests["output"] + private_tests["output"] + generated_tests["output"],
    }


def codecontests_rename_columns(dataset: datasets.Dataset) -> datasets.Dataset:
    dataset = dataset.map(
        lambda x: {
            "problem_id": x["problem_id"],
            "problem": x["description"],
            "test_cases": x["tests"],
            "difficulty": x["difficulty"],
            "source": x["source"],
            "language": "PYTHON3",
        }
    )

    return dataset


def apps_rename_columns(dataset: datasets.Dataset) -> datasets.Dataset:
    df = dataset.to_pandas()

    df = df.rename(
        columns={
            "question": "problem",
            "input_output": "test_cases",
            "difficulty": "difficulty",
            "problem_id": "problem_id",
            "name": "name",
            "language": "language",
            "source": "source",
        }
    )

    return datasets.Dataset.from_pandas(df)


def apps_process(dataset: datasets.Dataset, num_hf_proc_workers: int = 1) -> datasets.Dataset:
    dataset = dataset.map(
        lambda x: {
            "problem_id": compute_problem_id(x["question"]),
            "difficulty": x["difficulty"].upper(),
            "name": x.get("name") if x.get("name") else "UNKNOWN",
            "language": "PYTHON3",
            "source": (get_domain(x["url"]).replace("www.", "").replace(".com", "").upper() if x["url"] else "UNKNOWN_SOURCE"),
        },
        num_proc=num_hf_proc_workers,
    )

    dataset = dataset.filter(lambda x: filter_problem(x["question"]), num_proc=num_hf_proc_workers)
    dataset = dataset.filter(lambda x: filter_tests(x["input_output"]), num_proc=num_hf_proc_workers)

    dataset = dataset.map(
        lambda x: {
            "num_solutions": len(x["solutions"]),
        },
        num_proc=num_hf_proc_workers,
    )

    dataset = dataset.filter(lambda x: x["num_solutions"] > 0, num_proc=num_hf_proc_workers)

    dataset = apps_rename_columns(dataset)
    dataset = dataset.select_columns(COLUMNS)

    return dataset


def cps_groupby_problem_id(dataset: datasets.Dataset) -> datasets.Dataset:
    df = dataset.to_pandas()
    df = (
        df.groupby("problem_id")
        .agg(
            {
                "test_cases": list,
                "code": list,
                "name": "first",
                "description": "first",
            }
        )
        .reset_index()
    )

    return datasets.Dataset.from_pandas(df)


def rename_cps(dataset: datasets.Dataset) -> datasets.Dataset:
    df = dataset.to_pandas()
    df = df.rename(
        columns={
            "description": "problem",
            "tests": "test_cases",
            "difficulty": "difficulty",
            "source": "source",
            "problem_id": "problem_id",
            "name": "name",
        }
    )

    return datasets.Dataset.from_pandas(df)


def cps_process(dataset: datasets.Dataset, num_hf_proc_workers: int = 1) -> datasets.Dataset:
    dataset = dataset.filter(lambda x: x["verdict"] == "OK", num_proc=num_hf_proc_workers)

    dataset = dataset.map(
        lambda x: {
            "sample-tests": f"Sample Input\n{''.join(x['demo-input'])}\nSample Output\n{''.join(x['demo-output'])}",
        },
        num_proc=num_hf_proc_workers,
    )

    dataset = dataset.map(
        lambda x: {
            "description": x["problem-description"] + "\n" + x["input-specification"] + "\n" + x["output-specification"] + "\n" + x["sample-tests"],
            "problem_id": compute_problem_id(
                x["problem-description"] + "\n" + x["input-specification"] + "\n" + x["output-specification"] + "\n" + x["sample-tests"]
            ),
        },
        num_proc=num_hf_proc_workers,
    )

    dataset = cps_groupby_problem_id(dataset)

    dataset = dataset.filter(lambda x: filter_problem(x["description"]), num_proc=num_hf_proc_workers)

    dataset = dataset.map(
        lambda x: {
            "source": "CODEFORCES",
            "difficulty": "UNKNOWN",
            "test_cases": {
                "inputs": [i["input"] for i in x["test_cases"][0]],
                "outputs": [i["output"] for i in x["test_cases"][0]],
            },
            "language": "PYTHON3",
        },
        num_proc=num_hf_proc_workers,
    )

    dataset = dataset.filter(lambda x: filter_tests(x["test_cases"]), num_proc=num_hf_proc_workers)

    dataset = rename_cps(dataset)

    dataset = dataset.map(
        lambda x: {
            "solutions": x["code"],
            "num_solutions": len(x["code"]),
            "starter_code": "",
        },
        num_proc=num_hf_proc_workers,
    )

    dataset = dataset.select_columns(COLUMNS)

    # dump tests
    dataset = dataset.map(
        lambda x: {"test_cases": json.dumps(x["test_cases"])},
        num_proc=num_hf_proc_workers,
    )

    return dataset


def codecontests_process(dataset: datasets.Dataset, num_hf_proc_workers: int = 1) -> datasets.Dataset:
    """Process code contests dataset."""
    dataset = dataset.filter(lambda x: filter_problem(x["description"]), num_proc=num_hf_proc_workers)

    dataset = dataset.map(
        lambda x: {
            "problem_id": compute_problem_id(x["description"]),
            "source": codecontests_map_sources(x["source"]),
        },
        num_proc=num_hf_proc_workers,
    )

    dataset = dataset.map(
        lambda x: {
            "tests": codecontests_combine_tests(x["public_tests"], x["private_tests"], x["generated_tests"]),
            "num_solutions": len(x["solutions"]),
        },
        num_proc=num_hf_proc_workers,
    )

    dataset = dataset.filter(
        lambda x: filter_tests(x["tests"]),
        num_proc=num_hf_proc_workers,
    )

    dataset = dataset.filter(
        lambda x: filter_solutions(x["solutions"]),
        num_proc=num_hf_proc_workers,
    )

    dataset = dataset.filter(lambda x: filter_num_solutions(x["num_solutions"]), num_proc=num_hf_proc_workers)

    dataset = dataset.map(
        lambda x: {
            "starter_code": "",
        },
        num_proc=num_hf_proc_workers,
    )

    dataset = codecontests_rename_columns(dataset)
    dataset = dataset.select_columns(COLUMNS)

    # dump tests
    dataset = dataset.map(
        lambda x: {
            "test_cases": dump_tests(x["test_cases"]),
        },
        num_proc=num_hf_proc_workers,
    )

    return dataset


def standardize(dataset_name_or_path: str, num_hf_proc_workers: int = 1) -> datasets.Dataset:
    """Process code dataset.

    Args:
        dataset_name_or_path (str): Dataset name or path.
        num_hf_proc_workers (int): Number of Hugging Face processing workers.

    Returns:
        datasets.Dataset: Processed dataset.
    """
    dataset = datasets.load_dataset(
        dataset_name_or_path,
        split="all",
        trust_remote_code=True,
    )

    process_fn_map = {
        "code_contests": codecontests_process,
        "apps": apps_process,
        "taco": apps_process,
        "codeforces-python-submissions": cps_process,
    }

    dataset_name = dataset_name_or_path.split("/")[-1]

    process_fn = process_fn_map.get(dataset_name.lower())

    dataset = process_fn(dataset, num_hf_proc_workers)

    return dataset
