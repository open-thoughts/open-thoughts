# Training

We train [OpenThinker-7B](https://huggingface.co/open-thoughts/OpenThinker-7B) and [OpenThinker-32B](https://huggingface.co/open-thoughts/OpenThinker-32B) using [LLaMa-Factory](https://github.com/hiyouga/LLaMA-Factory).

We provide our training config files with the relevant hyperparameters:
- [`OpenThinker-7B.yaml`](./OpenThinker-7B.yaml)
- [`OpenThinker-32B.yaml`](./OpenThinker-32B.yaml)

These are same for the 7B and 32B models. Notably, we use `cutoff_len: 16384`. Some reasoning traces are longer than this. However we found that filtering out traces longer than this did not improve model performance.
