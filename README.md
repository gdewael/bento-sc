# express
scRNA-seq transformer benchmarking


## Todos
- [ ] Investigate numerical stability of zero truncated loss functions: https://github.com/pytorch/pytorch/blob/main/torch/_refs/nn/functional/__init__.py#L419
- [ ] MCV vs MLM
- [ ] Only non-zero vs random sampling vs proportional sampling vs only HVGs fixed
- [ ] Implement [CLS-level] tasks:
  - [ ] Molecular CV
  - [ ] Binomial split
  - [ ] Gaussian noise
  - [ ] Supervised celltype labeling
- [ ] Gather some initial baselines