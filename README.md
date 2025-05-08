<div align="center">

<img src="https://raw.githubusercontent.com/gdewael/bento-sc/refs/heads/main/assets/bento.svg" align="center" width="450" alt="bento-sc" href="https://github.com/gdewael/bento-sc">

<h1></h1>

BENchmarking Transformer-Obtained Single-Cell representations.


[![PyPi Version](https://img.shields.io/pypi/v/bento-sc.svg)](https://pypi.python.org/pypi/bento-sc/)
[![GitHub license](https://img.shields.io/github/license/gdewael/bento-sc)](https://github.com/gdewael/bento-sc/blob/main/LICENSE)
[![Documentation](https://readthedocs.org/projects/bento-sc/badge/?version=latest&style=flat-default)](https://bento-sc.readthedocs.io/en/latest/index.html)

</div>

## Single-cell language modeling

This repository is linked to the study called "A systematic assessment of single-cell language model configurations" ([preprint paper link](https://doi.org/10.1101/2025.04.02.646825)).

The package contains routines and definitions for pre-training single-cell (transcriptomic) language models.

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

If you end up using this code in your research, please cite:
```
@article {dewaele2025systematic,
	author = {De Waele, Gaetan and Menschaert, Gerben and Waegeman, Willem},
	title = {A systematic assessment of single-cell language model configurations},
	year = {2025},
	doi = {10.1101/2025.04.02.646825},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2025/04/08/2025.04.02.646825},
	journal = {bioRxiv}
}
```

## Accreditation original dataset authors

The following datasets were used:
- scTab (derived from CELLxGENE census) - CC-BY-4.0 License
- NeurIPS 2023 Perturbation data - CC-BY-4.0 License
- Replogle perturb-seq - CC-BY-4.0 License
- NeurIPS 2021 CITE-seq - CC-BY-4.0 License
- Circulating immune cells (CELLxGENE derived) - CC-BY-4.0 License
- Embryonic limb cell atlas - CC-BY-4.0 License
- Middle temporal gyrus among great apes - CC-BY-4.0 License
  
If you use *bento-sc*, please accredit the original authors:

scTab:
```
@article{fischer2024sctab,
	title={scTab: scaling cross-tissue single-cell annotation models},
	author={Fischer, Felix and Fischer, David S and Mukhin, Roman and Isaev, Andrey and Biederstedt, Evan and Villani, Alexandra-Chlo{\'e} and Theis, Fabian J},
	journal={Nature Communications},
	volume={15},
	number={1},
	pages={6611},
	year={2024},
	publisher={Nature Publishing Group UK London}
}
```

GRN Inference:
```
@article{szalata2025benchmark,
	title={A benchmark for prediction of transcriptomic responses to chemical perturbations across cell types},
	author={Sza{\l}ata, Artur and Benz, Andrew and Cannoodt, Robrecht and Cortes, Mauricio and Fong, Jason and Kuppasani, Sunil and Lieberman, Richard and Liu, Tianyu and Mas-Rosario, Javier and Meinl, Rico and others},
	journal={Advances in Neural Information Processing Systems},
	volume={37},
	pages={20566--20616},
	year={2025}
}
```

Post-perturbation prediction
```
@article{replogle2022mapping,
	title={Mapping information-rich genotype-phenotype landscapes with genome-scale Perturb-seq},
	author={Replogle, Joseph M and Saunders, Reuben A and Pogson, Angela N and Hussmann, Jeffrey A and Lenail, Alexander and Guna, Alina and Mascibroda, Lauren and Wagner, Eric J and Adelman, Karen and Lithwick-Yanai, Gila and others},
	journal={Cell},
	volume={185},
	number={14},
	pages={2559--2575},
	year={2022},
	publisher={Elsevier}
}
```

Surface protein abundance prediction:
```
@inproceedings{luecken2021sandbox,
	title={A sandbox for prediction and integration of DNA, RNA, and proteins in single cells},
	author={Luecken, Malte D and Burkhardt, Daniel Bernard and Cannoodt, Robrecht and Lance, Christopher and Agrawal, Aditi and Aliee, Hananeh and Chen, Ann T and Deconinck, Louise and Detweiler, Angela M and Granados, Alejandro A and others},
	booktitle={Thirty-fifth conference on neural information processing systems datasets and benchmarks track (Round 2)},
	year={2021}
}
```

Batch correction:
```
@article{zhang2023human,
	title={A human embryonic limb cell atlas resolved in space and time},
	author={Zhang, Bao and He, Peng and Lawrence, John EG and Wang, Shuaiyu and Tuck, Elizabeth and Williams, Brian A and Roberts, Kenny and Kleshchevnikov, Vitalii and Mamanova, Lira and Bolt, Liam and others},
	journal={Nature},
	pages={1--11},
	year={2023},
	publisher={Nature Publishing Group UK London}
}
```

```
@article{jorstad2023comparative,
	title={Comparative transcriptomics reveals human-specific cortical features},
	author={Jorstad, Nikolas L and Song, Janet HT and Exposito-Alonso, David and Suresh, Hamsini and Castro-Pacheco, Nathan and Krienen, Fenna M and Yanny, Anna Marie and Close, Jennie and Gelfand, Emily and Long, Brian and others},
	journal={Science},
	volume={382},
	number={6667},
	pages={eade9516},
	year={2023},
	publisher={American Association for the Advancement of Science}
}
```

```
@article{ivanova2023mrna,
	title={mRNA COVID-19 vaccine elicits potent adaptive immune response without the acute inflammation of SARS-CoV-2 infection},
	author={Ivanova, Ellie N and Shwetar, Jasmine and Devlin, Joseph C and Buus, Terkild B and Gray-Gaillard, Sophie and Koide, Akiko and Cornelius, Amber and Samanovic, Marie I and Herrera, Alberto and Mimitou, Eleni P and others},
	journal={Iscience},
	volume={26},
	number={12},
	year={2023},
	publisher={Elsevier}
}
```

