# Getting started

## Install

`bento-sc` is distribution on PyPI.
```bash
pip install bento-sc
```
Note: The package has been tested with `torch==2.2.2` and `pytorch-lightning==2.2.5`. If you encounter errors with `bento-sc` using more recent version of these two packages, consider downgrading.

You may need to [install PyTorch](https://pytorch.org/get-started/locally/) before running this command in order to ensure the right CUDA kernels for your system are installed.

## Download and prepare data

In order to get up and running, you will need to download the relevant data and process them to [`h5torch`-compatible HDF5 files](https://github.com/gdewael/h5torch).
All data downloading and processing routines are made available through a single CLI script `bentosc_data`:

```bash
usage: bentosc_data [-h] datafile

Data downloading script launching pad. Choose a datafile to download and process to h5torch format.

positional arguments:
  datafile    Datafile to download, choices: {scTab, scTab_upscaling, scTab_grn, neurips_citeseq, replogle_perturb, batchcorr_embryolimb, batchcorr_greatapes, batchcorr_circimm}

options:
  -h, --help  show this help message and exit
```

On each of the subcommands, you can also call `-h`. E.g.: `bentosc_data scTab -h`.

To download and preprocess the pre-training data:

```bash
bentosc_data scTab ./data_tmp/ ./scTab.h5t
```

To perform downstream task evaluations, additionally process any or all task-specific data:
```bash
bentosc_data scTab_upscaling ./scTab.h5t ./scTab_upsc_val.h5t val
bentosc_data scTab_upscaling ./scTab.h5t ./scTab_upsc_test.h5t test
bentosc_data scTab_grn ./scTab.h5t ./scTab_grn_val.h5t ./ext_pertdata.h5ad val
bentosc_data scTab_grn ./scTab.h5t ./scTab_grn_test.h5t /ext_pertdata.h5ad test
bentosc_data neurips_citeseq ./data_tmp/ ./scTab.h5t ./citeseq.h5t
bentosc_data replogle_perturb ./data_tmp/ ./scTab.h5t ./perturb.h5t
bentosc_data batchcorr_embryolimb ./data_tmp/ ./scTab.h5t ./batchcorr_el.h5t
bentosc_data batchcorr_greatapes ./data_tmp/ ./scTab.h5t ./batchcorr_ga.h5t
bentosc_data batchcorr_circimm ./data_tmp/ ./scTab.h5t ./batchcorr_ci.h5t
```

**Note:** downloading and processing all of this data will cumulatively take up quite a bit of time and storage space. Allow for at least 400Gb storage space.

## Pre-training a model

Pre-training can be performed through the CLI script `bentosc_pretrain`:
```bash
usage: bentosc_pretrain [-h] [--data_path str] [--lr float] [--ckpt_path str] [--tune_mode boolean] config_path logs_path

Pre-training script.

positional arguments:
  config_path          .yaml config file controlling most of the pre-training parameters.
  logs_path            Where to save tensorboard logs and model weight checkpoints.

options:
  -h, --help           show this help message and exit
  --data_path str      Data file. Overrides value in config file if specified (default: None)
  --lr float           Learning rate. Overrides value in config file if specified (default: None)
  --ckpt_path str      Continue from checkpoint (default: None)
  --tune_mode boolean  Don't pre-train whole model but run small experiment. (default: False)
```

The first input to `bentosc_pretrain`, a `config.yaml` file, controls most of the pre-training logic.
A minimal example of a working config file is:
```yaml
#DataModule args:
batch_size: 192 # per-gpu batch size
devices: [0, 1] # devices to use. here, the first and second GPU are used (total batch size: 384)
n_workers: 12 # number of CPUs to use in dataloading
in_memory: False # don't load scTab.h5t into memory fully, but read from disk during training.
val_sub: True # use the subsetted validation set of scTab.

# Data processing args:
return_zeros: False # if False, load in the cell profiles without zeros
allow_padding: False # if False, cut to min size in batch
input_processing: # cell-wise preprocessing functions
  - type: FilterTopGenes # get the top genes
    affected_keys: ["gene_counts", "gene_index", "gene_counts_true"]
    number: 1024
  - type: Bin # Bin input counts
    key: "gene_counts"
  - type: Mask # Mask 15% of input counts
    p: 0.15
    key: "gene_counts"
  - type: Bin # Bin output counts
    key: "gene_counts_true"

# Model args:
discrete_input: True # Set to True if using binned or rank input encodings
n_discrete_tokens: 29 # the number of bins
gate_input: False # use the gating mechanism on input embeddings
pseudoquant_input: False # use the pseudoquantization mechanism on input embeddings
dim: 512 # hidden dim of the transformer
depth: 10 # number of transformer encoder layers
dropout: 0.2 # dropout rate in attention matrix and FF
n_genes: 19331 # number of genes in dataset to initialize gene index vocabulary

# General learning args
lr: 3e-4 # Pre-training learning rate
train_on_all: False # Only compute loss on masked/noised positions
loss: # type of loss and parameters
  type: BinCE
  n_bins: 29 

# Pre-training args:
nce_loss: False # Use contrastive learning or not 
nce_dim: 64 # contrastive embedding dim
nce_temp: 1 # Temp in contrastive loss func

# Fine-tuning args:
celltype_clf_loss: False # Use celltype-clf, can be used during both pre-training and fine-tuning
modality_prediction_loss: False # Fine-tune for citeseq task
cls_finetune_dim: 164 # final linear layer projecting the CLS embedding. Should be 164 for scTab Celltype ID, and 134 for NeurIPS citeseq
perturb_mode: False # Fine-tune for perturbation task
```

Using this base design, one can train the "base" scLM configuration in our study.
Many more examples are available in our reproducibility GitHub repository.

## Evaluating performance on a downstream task

Once a model is pre-trained, its performance on downstream tasks can be evaluated.
Currently, `bento-sc` defines six downstream tasks.
Each one is implemented through a CLI script:

- Batch Correction: `bentosc_task_batchcorr`
- Celltype Identification: `bentosc_task_celltypeid`
- GRN Inference: `bentosc_task_grninfer`
- Post Perturbation Expression Prediction: `bentosc_task_perturb`
- Protein concentration Prediction: `bentosc_task_protconc` 
- Gene Expression upscaling: `bentosc_task_upscale`

The command line inputs for these scripts can be inspected via their `-h` flag, e.g.:
```bash
usage: bentosc_task_celltypeid [-h] [--data_path str] [--lr float] [--batch_size int] [--n_workers int] [--prefetch_factor int] [--tune_mode boolean] config_path checkpoint logs_path

Fine-tuning script for cell-type identification evaluation.

positional arguments:
  config_path           config_path
  checkpoint            checkpoint path
  logs_path             logs_path

options:
  -h, --help            show this help message and exit
  --data_path str       Data file. Overrides value in config file if specified (default: None)
  --lr float            Learning rate. Overrides value in config file if specified (default: None)
  --batch_size int      Batch size. Overrides value in config file if specified (default: None)
  --n_workers int       Num workers. Overrides value in config file if specified (default: None)
  --prefetch_factor int
                        Prefetch Factor of dataloader. Overrides value in config file if specified (default: None)
  --tune_mode boolean   Don't pre-train whole model but run small experiment. (default: False)
```

**Note:** many tasks require different config file values from their pre-training one.
For examples, refer to our reproducibility GitHub repository.
