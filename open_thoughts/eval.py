EVALUATION_DATASETS = {
    "HuggingFaceH4/MATH-500": {
        "eval_columns": ["problem"],
        "eval_splits": ["test"],
    },
    "Maxwell-Jia/AIME_2024": {
        "eval_columns": ["Problem"],
        "eval_splits": ["train"],
    },
    "AI-MO/aimo-validation-amc": {
        "eval_columns": ["problem"],
        "eval_splits": ["train"],
    },
    "livecodebench/code_generation_lite": {
        "eval_columns": ["question_content"],
        "eval_splits": ["test"],
    },
    "Idavidrein/gpqa": {
        "eval_columns": ["Question"],
        "eval_splits": ["train"],
        "eval_subset": "gpqa_diamond",
    },
}
