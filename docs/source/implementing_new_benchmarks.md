# Implementing a new benchmark

In principle, `bento-sc` can be extended towards any task that uses scRNA-seq data as input.
On this page, we will outline the steps needed to incorporate a new task in bento-sc.

In principle, this outline can also be used if you want to use pre-existing bento-sc scLMs for your own research tasks.
Note, however, that our codebase is primarily designed around (1) flexibility in scLM configuration design, and (2) benchmarking them on a specific task set.

## General steps

Roughly speaking, the steps to follow are as follows:
- For a task of interest, find a suitable dataset.
- Process the dataset to a [`h5torch`-compatible HDF5 file](https://github.com/gdewael/h5torch).
- Implement routines and/or a script fitting for said task


## Processing datasets to h5torch files

In all experiments, we use [`h5torch`-compatible HDF5 file](https://github.com/gdewael/h5torch).
The purpose of these files is to provide a similar interface as with `anndata` files, but allowing to data-loading from an on-disk format.
This is necessary, as it is impossible to load large corpora such as the scTab dataset all into memory.

To familiarize yourself with the `h5torch` syntax, take a look at its [docs](https://h5torch.readthedocs.io/en/latest/#package-concepts).
An example of how to obtain process an anndata `.h5ad` to a h5torch `.h5t` file, is given [here](https://github.com/gdewael/bento-sc/blob/7983cae5c512a99e54a096d893dafecd84e00247/bento_sc/utils/get_dataset.py#L1010)

## Implementing routines to load h5t files

Depending on your task, you may need to implement custom data loading functionalities.
All existing dataloading functionalities are located in [`bento_sc.data`](https://github.com/gdewael/bento-sc/blob/main/bento_sc/data.py).
The `config.yaml` files used for all tasks define how each cell is processed.
If your use case is not covered by the existing configuration possibilities, you will need to fork and extend our codebase.
All contributions are welcomed!
If you are not sure if your use case is covered, you are always welcome to open an issue.

To explain the structure of [`bento_sc.data`](https://github.com/gdewael/bento-sc/blob/main/bento_sc/data.py) briefly: `bento-sc` uses `BentoDataModule` as a central data loading object, in which the training, validation, and testing fraction are located under `.train`, `.val`, and `.test` of the instantiated class objects, respectively.
All datasets are [`h5torch.Dataset` objects](https://h5torch.readthedocs.io/en/latest/h5torch.html#h5torch.dataset.Dataset).
Datasets contain a `sample_processor` function, which modulates how each sample (cell / row in the `.h5t` file) is processed before it is returned as a batch.
The default sample_processor function is [`CellSampleProcessor`](https://github.com/gdewael/bento-sc/blob/7983cae5c512a99e54a096d893dafecd84e00247/bento_sc/data.py#L392).

