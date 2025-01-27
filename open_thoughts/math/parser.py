from math_verify.metric import math_metric
from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig

llm_judge = False
gold_is_latex = True
verify_func = math_metric(
    gold_extraction_target=(LatexExtractionConfig() if gold_is_latex else ExprExtractionConfig(),),
    pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()),
    aggregation_function=max,
    fallback_mode="first_match",
    precision=6,
)


def validate_solution(row):
    """Validate a single solution using the verification function."""
    extracted_answers = ""  # Initialize as empty string instead of None
    gold_answers = ""  # Initialize as empty string instead of None
    grade = 0
    try:
        # Use the verification function
        grade, extracted_answers = verify_func([row["ground_truth_solution"]], [row["deepseek_solution"]])

        if extracted_answers is None:
            extracted_answers = ""  # Use empty string instead of None
            gold_answers = ""  # Use empty string instead of None
        else:
            gold_answers = str(extracted_answers[0])  # Convert to string
            extracted_answers = str(extracted_answers[1])  # Convert to string

        return {
            **row,  # Keep all existing fields
            "extracted_answer": extracted_answers,
            "extracted_gold": gold_answers,
            "verifier_label": grade == 1,
            "error": "",  # Empty string instead of None
        }

    except Exception as e:
        return {
            **row,  # Keep all existing fields
            "extracted_answer": extracted_answers,
            "extracted_gold": gold_answers,
            "verifier_label": grade == 1,
            "error": str(e),
        }


def parser(ds):
    validated_results = ds.map(validate_solution)
    return validated_results
