# Data Generation

Set up the environment and HF as outlined in the parent README.

Set the DeepSeek API key:
```
export DEEPSEEK_API_KEY=your_api_key
```

Set HF_ORG to your organization id. Set HF_PRIVATE=true if you want to push to a private repo.
```
export HF_ORG=your_org_id
export HF_PRIVATE=false
```


Cached responses make iteration time faster. If you want to regenerate without caching, set the following. 
```
export CURATOR_DISABLE_CACHE=true
```

## Domains

First run with the `--dry-run` flag to run on a small subset of the data.

### Math

This will push the math dataset (along with intermediate datasets) to the HuggingFace dataset `{HF_ORG}/open-thoughts-math`.

```
python open_thoughts/math/maths.py --dry-run
```

### Code

This will push the code dataset (along with intermediate datasets) to the HuggingFace dataset `{HF_ORG}/open-thoughts-code`.

**WARNING**: This code when run without `--dry-run` uses the `deepmind/code_contests` [dataset](https://huggingface.co/datasets/deepmind/code_contests), which is >100 GB. Make sure you have enough disk space.

```
python open_thoughts/code/code.py --dry-run
```

### Science

This will push the science dataset (along with intermediate datasets) to the HuggingFace dataset `{HF_ORG}/open-thoughts-science`.

```
python open_thoughts/science/science.py --dry-run
```

### Puzzle

This will push the puzzle dataset (along with intermediate datasets) to the HuggingFace dataset `{HF_ORG}/open-thoughts-puzzle`.

```
python open_thoughts/puzzle/puzzle.py --dry-run
```


## Combine and verify

After running the above, you can combine the datasets and run verification using:

```
python open_thoughts/mix.py --dry-run
```


## Source Datasets

### Math

1. [AI-MO/NuminaMath-7B-CoT](https://huggingface.co/AI-MO/NuminaMath-7B-CoT)

### Code

1. [MatrixStudio/Codeforces-Python-Submissions](https://huggingface.co/datasets/MatrixStudio/Codeforces-Python-Submissions)
2. [BAAI/TACO](https://huggingface.co/datasets/BAAI/TACO)
3. [codeparrot/apps](https://huggingface.co/datasets/codeparrot/apps)
4. [deepmind/code_contests](https://huggingface.co/datasets/deepmind/code_contests)

### Science 

1. [camel-ai/physics](https://huggingface.co/datasets/camel-ai/physics)
2. [camel-ai/chemistry](https://huggingface.co/datasets/camel-ai/chemistry)
3. [camel-ai/biology](https://huggingface.co/datasets/camel-ai/biology)


### Puzzle

1. [INK-USC/riddle_sense](https://huggingface.co/datasets/INK-USC/riddle_sense)
