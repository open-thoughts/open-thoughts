
<!-- markdownlint-disable first-line-h1 -->
<!-- markdownlint-disable html -->
<!-- markdownlint-disable no-duplicate-header -->

<div align="center">
  <img src="open_thoughts.png" width="60%" alt="Open Thoughts GitHub Repository" />
</div>
<p align="center">
  <a href="https://open-thoughts.ai">
    <img alt="Static Badge" src="https://img.shields.io/badge/Home-open--thoughts.ai-blue?style=flat&link=https%3A%2F%2Fopen-thoughts.ai">
  </a>
  <a href="https://huggingface.co/open-thoughts">
    <img alt="Hugging Face" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Open%20Thoughts-blue?color=ffc107&logoColor=white&style=flat&link=https%3A%2F%2Fhuggingface.co/open-thoughts">
  </a>
  <br>
  <i>Curating the best open reasoning datasets</i><br> 
  A collaboration led by <a href="https://bespokelabs.ai/">bespokelabs</a> and the <a href="https://www.datacomp.ai/">datacomp</a> community

</p>
<hr>


Our first goal is to curate a reasoning dataset to train state of the art small reasoning models that surpass [DeepSeek-R1-Distill-32B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B) and [DeepSeek-R1-Distill-7B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B) on math and code reasoning benchmarks.

# News
- **[2025/01/28]** 🎉 [Open Thoughts](https://www.open-thoughts.ai/) launches with [open-thoughts-114k dataset](https://huggingface.co/datasets/open-thoughts/open-thoughts-114k) and [open-thinker-7B model](https://huggingface.co/open-thoughts/open-thinker-7B).
- **[2025/01/22]** 🎉 We [release](https://www.bespokelabs.ai/blog/bespoke-stratos-the-unreasonable-effectiveness-of-reasoning-distillation) our [Bespoke-Stratos-17k dataset](https://huggingface.co/datasets/bespokelabs/Bespoke-Stratos-17k) and [Bespoke-Stratos-32B model](https://huggingface.co/bespokelabs/Bespoke-Stratos-32B) 

# Links
- 📊 Open Thoughts Launch Blog Post
- 🧠 [open-thoughts-114k dataset](https://huggingface.co/datasets/open-thoughts/open-thoughts-114k)
- 🤖 [open-thinker-7B model](https://huggingface.co/open-thoughts/open-thinker-7B)
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

Currently, we are generating data for the following domains:
1. Code
2. Math
3. Science
4. Puzzle

The recipe is outlined below:
<picture>
    <source media="(prefers-color-scheme: light)" width="100%" srcset="diagram.png">
    <img alt="Data Curation Recipe" width="100%" src="diagram_dark.png">
</picture>

More instructions are in [open_thoughts/README.md](open_thoughts/README.md).


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

We are a team of researchers and engineers from [Bespoke Labs](https://www.bespokelabs.ai/), Stanford, University of California Berkeley, University of Washington, Juelich Supercomputing Center (JSC), LAION, UCLA, UNC Chapel Hill, UT Austin, and Toyota Research Institute united around building the best datasets (and thus the best models). See our previous works at [datacomp.ai](https://www.datacomp.ai/) and [mlfoundations](https://github.com/mlfoundations).

# Sponsors
Open Thoughts is supported by 
- [Bespoke Labs](https://www.bespokelabs.ai/)
- [Lambda Labs](https://lambdalabs.com/)
- [IFML](https://www.ifml.institute/)
- [NSF](https://www.nsf.gov/)
- [Juelich Supercomputing Center](https://www.fz-juelich.de/en/ias/jsc)
- [Toyota Research Institute](https://www.tri.global/)
