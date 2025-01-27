def mocked_reasoner(ds, answer_column: str = None):
    reasoning = "Deep Thought is thinking for 7.5 million years."
    if answer_column:
        ds = ds.map(lambda x: {"deepseek_solution": x[answer_column], "ground_truth_solution": x[answer_column]})
    else:
        solution = "The answer is 42."
        ds = ds.add_column("deepseek_solution", [solution] * len(ds))
    ds = ds.add_column("reasoning", [reasoning] * len(ds))
    return ds
