# express
scRNA-seq transformer benchmarking


## Todos
- [x] Investigate numerical stability of zero truncated loss functions: https://github.com/pytorch/pytorch/blob/main/torch/_refs/nn/functional/__init__.py#L419
- [x] MCV vs MLM
- [x] Only non-zero vs random sampling vs only HVGs fixed
- [x] Implement [CLS-level] tasks:
  - [x] Molecular CV
  - [x] Binomial split
  - [x] Gaussian noise
  - [x] Supervised celltype labeling
- [ ] Gather some initial baselines

### Baseline list
- [ ] Celltype classification on scTab + 5 datasets from [this paper](https://www.biorxiv.org/content/10.1101/2024.02.16.580624v1.full.pdf)
  - [ ] VERSUS (1) unpre-trained Transformer same arch
  - [ ] VERSUS (2) log reg
  - [ ] VERSUS (3) NN
  - [ ] VERSUS (4) NN[pre-trained VAE] scVI method
- [ ] "zero-shot" Celltype clustering on scTab + 5 datasets from [this paper](https://www.biorxiv.org/content/10.1101/2024.02.16.580624v1.full.pdf). For procedure/metrics see [this paper](https://www.biorxiv.org/content/10.1101/2023.10.16.561085v2.full.pdf).
  - [ ] VERSUS (1) PCA 
  - [ ] VERSUS (2) VAE dataset specific
- [ ] Batch Integration on Pancreas dataset see [this paper](https://www.biorxiv.org/content/10.1101/2023.10.16.561085v2.full.pdf)
  - [ ] VERSUS? ...
  - [ ] VERSUS? ...
  - [ ] CHECK THIS https://openproblems.bio/results/batch_integration_embed/ scIB score
- [ ] multiome prediction as in [this paper](https://www.biorxiv.org/content/10.1101/2024.02.16.580624v1.full.pdf)
  - [ ] VERSUS (1) unpre-trained Transformer same arch
  - [ ] VERSUS (2) log reg
  - [ ] VERSUS (3) NN
  - [ ] VERSUS (4) NN[pre-trained VAE] scVI method
  - [ ] ALSO CHECK THIS https://openproblems.bio/events/2022-08_neurips/

- [ ] Perturbation prediction - scGPT paper does this but weirdly -- check how GEARS does it?
  - [ ] siamese network cell 1 input cell 2 input --> prediction for each gene??
    - [ ] Do this be comparing gene embeddings or cell embeddings?
  - CHECK THIS https://www.kaggle.com/competitions/open-problems-single-cell-perturbations/overview/evaluation
 
Make a note on GEX reconstruction: as this is essentially what we are optimizing for, we exclude this.