from datasets import load_dataset


def filter_problems(x):
    for keyword in ["figure", "diagram", "jpeg", "png", "jpg", "svg", "answer:"]:
        if keyword in x["problem"].lower():
            return False
    if x["problem"].lower().startswith("a)") and "b)" in x["problem"].lower():  # These are multipart questions
        return False
    if x["solution"] is None:
        return False
    if x["solution"] == "":
        return False
    if "\\boxed{}" in x["solution"].lower():  # This is QED, so these are proofs
        return False
    if "\\boxed{" not in x["solution"].lower():
        return False
    return True


if __name__ == "__main__":
    numina = load_dataset("AI-MO/NuminaMath-CoT", split="train")
    stats = []
    for source in ["amc_aime", "olympiads", "aops_forum", "math"]:
        for keyword in ["figure", "diagram", "jpeg", "png", "jpg", "svg", "answer:", "\\boxed{}"]:
            print(f"##### {source} #####")
            ds = numina.filter(lambda x: x["source"] == source)
            ds = ds.filter(lambda x: keyword in x["problem"])
            if len(ds) > 0:
                ds.add_column("filter", [keyword] * len(ds))
            ds.push_to_hub(f"mlfoundations-dev/{source}_{keyword}")
            stats.append(f"{keyword}: {len(ds)} https://huggingface.co/datasets/mlfoundations-dev/{source}_{keyword}")
        ds = numina.filter(filter_problems)
        print("\n".join(stats))
        print(f"original {source}: ", len(numina))
        print(f"filtered {source}: ", len(ds))


"""
##### amc_aime #####
figures:  132 https://huggingface.co/datasets/mlfoundations-dev/amc_aime_figures
diagrams:  39 https://huggingface.co/datasets/mlfoundations-dev/amc_aime_diagrams
links:  258 https://huggingface.co/datasets/mlfoundations-dev/amc_aime_links
imgs:  27 https://huggingface.co/datasets/mlfoundations-dev/amc_aime_imgs
multipart:  0 https://huggingface.co/datasets/mlfoundations-dev/amc_aime_multipart
answer:  0 https://huggingface.co/datasets/mlfoundations-dev/amc_aime_answer
grid:  30 https://huggingface.co/datasets/mlfoundations-dev/amc_aime_grid
plot:  8 https://huggingface.co/datasets/mlfoundations-dev/amc_aime_plot
empty_boxed:  0 https://huggingface.co/datasets/mlfoundations-dev/amc_aime_empty_boxed
no_boxed:  145 https://huggingface.co/datasets/mlfoundations-dev/amc_aime_no_boxed
no_solution:  0 https://huggingface.co/datasets/mlfoundations-dev/amc_aime_no_solution
original amc_aime:  4070
filtered amc_aime:  3736

##### olympiads #####
figures:  4075 https://huggingface.co/datasets/mlfoundations-dev/olympiads_figures
diagrams:  1859 https://huggingface.co/datasets/mlfoundations-dev/olympiads_diagrams
links:  422 https://huggingface.co/datasets/mlfoundations-dev/olympiads_links
imgs:  421 https://huggingface.co/datasets/mlfoundations-dev/olympiads_imgs
multipart:  1078 https://huggingface.co/datasets/mlfoundations-dev/olympiads_multipart
answer:  150 https://huggingface.co/datasets/mlfoundations-dev/olympiads_answer
grid:  2041 https://huggingface.co/datasets/mlfoundations-dev/olympiads_grid
plot:  226 https://huggingface.co/datasets/mlfoundations-dev/olympiads_plot
empty_boxed:  9650 https://huggingface.co/datasets/mlfoundations-dev/olympiads_empty_boxed
no_boxed:  10764 https://huggingface.co/datasets/mlfoundations-dev/olympiads_no_boxed
no_solution:  0 https://huggingface.co/datasets/mlfoundations-dev/olympiads_no_solution
original olympiads:  150563
filtered olympiads:  122742

##### aops_forum #####
figures:  397 https://huggingface.co/datasets/mlfoundations-dev/aops_forum_figures
diagrams:  144 https://huggingface.co/datasets/mlfoundations-dev/aops_forum_diagrams
links:  619 https://huggingface.co/datasets/mlfoundations-dev/aops_forum_links
imgs:  338 https://huggingface.co/datasets/mlfoundations-dev/aops_forum_imgs
multipart:  107 https://huggingface.co/datasets/mlfoundations-dev/aops_forum_multipart
answer:  1 https://huggingface.co/datasets/mlfoundations-dev/aops_forum_answer
grid:  344 https://huggingface.co/datasets/mlfoundations-dev/aops_forum_grid
plot:  23 https://huggingface.co/datasets/mlfoundations-dev/aops_forum_plot
empty_boxed:  8 https://huggingface.co/datasets/mlfoundations-dev/aops_forum_empty_boxed
no_boxed:  11789 https://huggingface.co/datasets/mlfoundations-dev/aops_forum_no_boxed
no_solution:  0 https://huggingface.co/datasets/mlfoundations-dev/aops_forum_no_solution
original aops_forum:  30192
filtered aops_forum:  17755

##### math #####
figures:  91 https://huggingface.co/datasets/mlfoundations-dev/math_figures
diagrams:  93 https://huggingface.co/datasets/mlfoundations-dev/math_diagrams
links:  0 https://huggingface.co/datasets/mlfoundations-dev/math_links
imgs:  0 https://huggingface.co/datasets/mlfoundations-dev/math_imgs
multipart:  0 https://huggingface.co/datasets/mlfoundations-dev/math_multipart
answer:  0 https://huggingface.co/datasets/mlfoundations-dev/math_answer
grid:  48 https://huggingface.co/datasets/mlfoundations-dev/math_grid
plot:  25 https://huggingface.co/datasets/mlfoundations-dev/math_plot
empty_boxed:  0 https://huggingface.co/datasets/mlfoundations-dev/math_empty_boxed
no_boxed:  0 https://huggingface.co/datasets/mlfoundations-dev/math_no_boxed
no_solution:  0 https://huggingface.co/datasets/mlfoundations-dev/math_no_solution
original math:  7477
filtered math:  7289

"""
