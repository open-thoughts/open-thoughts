sources = [
    "UNKNOWN_SOURCE",
    "CODECHEF",
    "CODEFORCES",
    "HACKEREARTH",
    "CODEJAM",
    "ATCODER",
    "AIZU",
]

code_contests_sources_map = {str(i): source for i, source in enumerate(sources)}

languges = ["UNKNOWN_LANGUAGE", "PYTHON", "CPP", "PYTHON3", "JAVA"]

code_contests_languages_map = {str(i): language for i, language in enumerate(languges)}

COLUMNS = [
    "problem_id",
    "name",
    "problem",
    "solutions",
    "test_cases",
    "difficulty",
    "language",
    "source",
    "num_solutions",
    "starter_code",
]
