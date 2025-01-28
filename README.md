# Open Thoughts
**Curating the best open reasoning datasets.**  A [bespokelabs](https://bespokelabs.ai/) and [datacomp](https://www.datacomp.ai/) effort.

Our first goal is to curate a reasoning dataset to train a model that outperforms DeepSeek-R1-Distill [32B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B) and [7B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B).

# News
- **[2025/01/28]** 🎉 Launch of the *Open Thoughts* project and [OpenThoughts-114k dataset](https://huggingface.co/mlfoundations-dev)
- **[2025/01/22]** 🎉 We [release](https://www.bespokelabs.ai/blog/bespoke-stratos-the-unreasonable-effectiveness-of-reasoning-distillation) our [Bespoke-Stratos-17k dataset](https://huggingface.co/datasets/bespokelabs/Bespoke-Stratos-17k) and [Bespoke-Stratos-32B model](https://huggingface.co/bespokelabs/Bespoke-Stratos-32B) 

# Links
- 📊 [Bespoke-Stratos Blog Post](https://www.bespokelabs.ai/blog/bespoke-stratos-the-unreasonable-effectiveness-of-reasoning-distillation)
- 🧠 [Bespoke-Stratos-17k dataset](https://huggingface.co/datasets/bespokelabs/Bespoke-Stratos-17k)
- 🤖[Bespoke-Stratos-32B model](https://huggingface.co/bespokelabs/Bespoke-Stratos-32B)
- 🤖 [Bespoke-Stratos-7B model](https://huggingface.co/bespokelabs/Bespoke-Stratos-7B)

# Installation
```
make install
poetry shell
```
Set the DeepSeek API key:
```
export DEEPSEEK_API_KEY=your_api_key
```

Set HF_ORG to your organization id. Set HF_PRIVATE=true if you want to push to a private repo.
```
export HF_ORG=your_org_id
export HF_PRIVATE=false
```

# Data Generation
More instructions in [open_thoughts/README.md](open_thoughts/README.md).

Currently, we are generating data for the following domains:
1. Code
2. Math
3. Science
4. Puzzle

# Training and Evaluation
Training and evaluation code coming soon.

# Results

||Bespoke-Stratos-7B|Qwen2.5-7B-Instruct|DeepSeek-R1-Distill-Qwen-7B (Ours / Reported)| Open-Thinker-7B |
|---|---|---|---|--- |
|AIME2024|20.0|10.0|43.3 / 55.5| ? |
|MATH500|82.0|74.2|89.4 / 92.8| ? |
|GPQA-Diamond|37.8|33.3|44.9 / 49.1| ? |
|LiveCodeBench v2 Easy|71.4|65.9|81.3 /-| ? |
|LiveCodeBench v2 Medium|25.5|18.9|42.2 / -| ? |
|LiveCodeBench v2 Hard|1.6|3.3|2.4 / - | ? |
|LiveCodeBench v2 All|36.1|31.9|46.6 / -  | ? |

# About Us

We are a team of researchers and engineers from Bespoke Labs, Stanford, University of California Berkeley, University of Washington, Juelich Supercomputing Center (JSC), LAION, UCLA, UNC Chapel Hill, and Toyota Research Institute united around building the best datasets (and thus the best models). See our previous works at [datacomp.ai](https://www.datacomp.ai/).

# Sponsors
Open Thoughts is supported by 
- [Bespoke Labs](https://www.bespokelabs.ai/)
- [Lambda Labs](https://lambdalabs.com/)
- [Texas Advanced Computing Center](https://tacc.utexas.edu/)
- [Juelich Supercomputing Center](https://www.fz-juelich.de/en/ias/jsc)
- [Toyota Research Institute](https://www.tri.global/)
