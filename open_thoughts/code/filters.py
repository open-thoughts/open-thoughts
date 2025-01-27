import ast
import json
from typing import Dict, List


def filter_problem(description: str, min_description_length: int = 200) -> bool:
    if "http://" in description.lower():
        return False
    if "[image]" in description.lower():
        return False
    if len(description) < min_description_length:
        return False
    return True


def filter_tests(tests: Dict[str, List[Dict[str, List[str]]]]) -> bool:
    if isinstance(tests, str):
        try:
            tests = json.loads(tests)
        except:
            try:
                tests = ast.literal_eval(tests)
            except:
                tests = None
    if tests is None:
        return False
    if len(tests.get("inputs", [])) == 0:
        return False
    if len(tests.get("outputs", [])) == 0:
        return False
    return True


def filter_solutions(solutions: Dict[str, List[str]]) -> bool:
    if solutions is None:
        return False
    if isinstance(solutions, str):
        solutions = json.loads(solutions)
    if isinstance(solutions, list):
        if len(solutions) == 0:
            return False
    elif isinstance(solutions, dict):
        if len(solutions.get("solution", [])) == 0:
            return False
    return True


def filter_num_solutions(num_solutions: int) -> bool:
    return num_solutions > 0
