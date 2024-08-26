# bento-sc

BENchmarking Transformer-Obtained Single-Cell embeddings


## Todos
- [ ] fix `process_cxg_subset.py` to use data saved in package name
- [ ] 

### Baseline list
- (1) Celltype classification on scTab test set
  - Versus: log reg, small MLP, .. (unpre-trained transformer), pre-trained MLP (cfr scVI)

- (2) "Zero-shot" Celltype clustering + batch integration on great apes - human dataset (cfr. [this paper](https://www.biorxiv.org/content/10.1101/2024.02.16.580624v1.full.pdf)). 
  - Optionally, gather more datasets
  - For procedure/metrics see [this paper](https://www.biorxiv.org/content/10.1101/2023.10.16.561085v2.full.pdf).
  - Versus: PCA (on HVG or not), scVI/VAE, scVI trained on CxG

- (2) Modality prediction
  - NeurIPS 2021 Cite-seq ships with a split consisting of a test set from different donor measured at a different site.
  - Training set-up:
    - Standard [CLS] token embedding to predicting the modality
  - Citation: Neurips comp dataset paper and [this paper](https://www.biorxiv.org/content/10.1101/2024.02.16.580624v1.full.pdf)
  - Versus: log reg, small MLP, .. (unpre-trained transformer)


- (3) Perturbation prediction
  - Replogle K562 Essential `"K562_essential_raw_singlecell_01.h5ad"`
  - Filter genes that are in cellxgene census pre-training set
  - Filter samples that correspond to perturbations of genes that are observed in both the cellxgene census pre-training set and the fine-tuning set.
  - Training set-up:
    - Split random perturbations to val and test set
    - For these perturbations, use held-out set of control samples (one set for val and test)
    - For training: randomly pair control samples to perturbed sample
    - Predicted gene exp is always LogP1
    - TODO: Tune embedding uniform init 
  - Citation: GEARS and scGPT.
  - Versus: Log reg, small MLP, dual encoder network, .. (unpre-trained transformer)

 
- What about GEX reconstruction?: as this is essentially what we are optimizing for, we exclude this.


## Notes on code structure
All data are loaded in as counts via `h5torch`-compatible HDF5 files. In the Dataset object, the cell measurements are preprocessed using custom class objects, which can be extended with any kind of per-sample preprocessing.

## Trained models
1. Binned + pred Bin
2. Norm per count + pred Bin
3. Norm per cell + pred Bin
4. Norm per count + logp1 + pred Bin
5. Norm per cell + logp1 + pred Bin
6. Count Rank + pred Bin
7. Count Rank + mask/pred Rank
8. Count Rank + mask/pred Gene_id
