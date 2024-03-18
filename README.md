# express
scRNA-seq transformer benchmarking


## Todos
- [x] Investigate numerical stability of zero truncated loss functions: https://github.com/pytorch/pytorch/blob/main/torch/_refs/nn/functional/__init__.py#L419
- [x] MCV vs MLM
- [ ] Only non-zero vs random sampling vs only HVGs fixed
- [x] Implement [CLS-level] tasks:
  - [x] Molecular CV
  - [x] Binomial split
  - [x] Gaussian noise
  - [x] Supervised celltype labeling
- [ ] Gather some initial baselines