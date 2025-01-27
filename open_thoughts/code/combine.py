from datasets import Sequence, Value, concatenate_datasets


def combine(datasets_dict):
    apps = datasets_dict["codeparrot/apps"]
    code_contests = datasets_dict["deepmind/code_contests"]
    taco = datasets_dict["BAAI/TACO"]
    codeforces = datasets_dict["MatrixStudio/Codeforces-Python-Submissions"]
    # Standardize schema for all datasets
    apps = apps.map(lambda x: {"language": [x["language"]] if isinstance(x["language"], str) else x["language"]["language"]})

    taco = taco.map(lambda x: {"language": [x["language"]] if isinstance(x["language"], str) else x["language"]})
    code_contests = code_contests.map(lambda x: {"language": [x["language"]] if isinstance(x["language"], str) else x["language"]})
    codeforces = codeforces.map(lambda x: {"language": [x["language"]] if isinstance(x["language"], str) else x["language"]})
    # codeforces = codeforces.remove_columns(["__index_level_0__"])
    codeforces = codeforces.remove_columns(["solutions"])
    apps = apps.remove_columns(["solutions"])
    taco = taco.remove_columns(["solutions"])
    code_contests = code_contests.remove_columns(["solutions"])
    code_contests = code_contests.cast_column("difficulty", Value("string"))
    code_contests = code_contests.cast_column("source", Value("string"))

    new_features = code_contests.features
    new_features["difficulty"] = Value("string")
    new_features["language"] = Sequence(Value("string"))
    code_contests = code_contests.cast(new_features)
    apps = apps.cast(new_features)
    taco = taco.cast(new_features)
    codeforces = codeforces.cast(new_features)

    # Concatenate all datasets
    code_stratos_scale = concatenate_datasets([code_contests, apps, taco, codeforces])

    print(f"Total examples: {len(code_stratos_scale)}")

    return code_stratos_scale
