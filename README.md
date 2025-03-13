<div align="center">

<img src="https://raw.githubusercontent.com/gdewael/bento-sc/refs/heads/main/assets/bento.svg" align="center" width="450" alt="bento-sc">

<h1></h1>

BENchmarking Transformer-Obtained Single-Cell representations.

[![PyPi Version](https://img.shields.io/pypi/v/bento-sc.svg)](https://pypi.python.org/pypi/bento-sc/)
[![GitHub license](https://img.shields.io/github/license/gdewael/bento-sc)](https://github.com/gdewael/bento-sc/blob/main/LICENSE)
[![Documentation](https://readthedocs.org/projects/bento-sc/badge/?version=latest&style=flat-default)](https://bento-sc.readthedocs.io/en/latest/index.html)

</div>

## Single-cell language modeling

This package contains routines and definitions for pre-training single-cell (transcriptomic) language models.

Package features:
- Memory-efficient scRNA-seq dataloading from [`h5torch`-compatible HDF5 files](https://github.com/gdewael/h5torch).
- `yaml`-configurable language model training scripts.
- Modular and extendable data preprocessing pipelines.
- A diverse set of downstream tasks to evaluate scLM performance.
- Full reproducibility instructions of our study results via [bento-sc-reproducibility](https://github.com/gdewael/bento-sc-reproducibility).



## Install

`bento-sc` is distributed on PyPI.
```bash
pip install bento-sc
```
Note: The package has been tested with `torch==2.2.2` and `pytorch-lightning==2.2.5`. If you encounter errors with `bento-sc` using more recent version of these two packages, consider downgrading.

You may need to [install PyTorch](https://pytorch.org/get-started/locally/) before running this command in order to ensure the right CUDA kernels for your system are installed.

## Package usage and structure 

Please refer to our [documentation page](https://bento-sc.readthedocs.io/en/latest/index.html).

## Academic reproducibility

All config files and scripts that were used to pre-train models and fine-tune them towards downstream tasks are included in a separate GitHub repository: [bento-sc-reproducibility](https://github.com/gdewael/bento-sc-reproducibility).

In addition, all scripts to reproduce the "baselines" in our study are located in the [bento-sc-reproducibility](https://github.com/gdewael/bento-sc-reproducibility) repository.

## Citation

:eyes: :eyes: :eyes: