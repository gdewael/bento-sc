import os
from os.path import join
import anndata
import pandas as pd
import numpy as np
from math import ceil
from tqdm import tqdm
import h5torch
from scipy.sparse import csr_matrix
import h5py

BASE_PATH = "../data/"

files = [
    join(BASE_PATH, file) for file 
    in sorted(os.listdir(BASE_PATH), key=lambda x: int(x.split('.')[0])) 
    if file.endswith('.h5ad')
]


def read_obs(path):
    obs = anndata.read_h5ad(path, backed='r').obs
    obs['tech_sample'] = obs.dataset_id.astype(str) + '_' + obs.donor_id.astype(str)
    return obs

# read obs
print('Loading obs...')
obs = pd.concat([read_obs(file) for file in files]).reset_index(drop=True)
for col in obs.columns:
    if obs[col].dtype == object:
        obs[col] = obs[col].astype('category')
        obs[col].cat.remove_unused_categories()



def get_split(samples, val_split: float = 0.15, test_split: float = 0.15, seed=1):
    rng = np.random.default_rng(seed=seed)

    samples = np.array(samples)
    rng.shuffle(samples)
    n_samples = len(samples)

    n_samples_val = ceil(val_split * n_samples)
    n_samples_test = ceil(test_split * n_samples)
    n_samples_train = n_samples - n_samples_val - n_samples_test

    return {
        'train': samples[:n_samples_train],
        'val': samples[n_samples_train:(n_samples_train + n_samples_val)],
        'test': samples[(n_samples_train + n_samples_val):]
    }


def subset(splits, frac):
    assert 0. < frac <= 1.
    if frac == 1.:
        return splits
    else:
        return splits[:ceil(frac * len(splits))]


splits = {'train': [], 'val': [], 'test': []}
tech_sample_splits = get_split(obs.tech_sample.unique().tolist())
for x in ['train', 'val', 'test']:
    # tech_samples are already shuffled in the get_split method -> just subselect to subsample donors
    if x == 'train':
        # only subset training data set
        splits[x] = obs[obs.tech_sample.isin(subset(tech_sample_splits[x], 1.))].index.to_numpy()
    else:
        splits[x] = obs[obs.tech_sample.isin(tech_sample_splits[x])].index.to_numpy()

assert len(np.intersect1d(splits['train'], splits['val'])) == 0
assert len(np.intersect1d(splits['train'], splits['test'])) == 0
assert len(np.intersect1d(splits['val'], splits['train'])) == 0
assert len(np.intersect1d(splits['val'], splits['test'])) == 0

rng = np.random.default_rng(seed=1)

splits['train'] = rng.permutation(splits['train'])
splits['val'] = rng.permutation(splits['val'])
splits['test'] = rng.permutation(splits['test'])

splits2 = {}
splits2["train"] = splits["train"][:(len(splits["train"]) // 1024) * 1024]
splits2["val"] = splits["val"][:(len(splits["val"]) // 1024) * 1024]
splits2["test"] = splits["test"][:(len(splits["test"]) // 1024) * 1024]



len_ = len(obs)

f_out = h5torch.File("../cellxgene.h5t", "w")

f = h5py.File("../data/0.h5ad")
mat = csr_matrix((f["X/data"], f["X/indices"], f["X/indptr"]), shape=(f["X/indptr"].shape[0]-1, 19331))
mat.indices = mat.indices.astype("int16")
f_out.register(
    mat,
    axis="central",
    mode = "csr",
    dtype_save="float32",
    dtype_load="float32",
    csr_load_sparse=True,
    length=len_,
)
f.close()


for i in tqdm(range(1, 100)):
    f = h5py.File("../data/%s.h5ad" % i)
    mat = csr_matrix((f["X/data"], f["X/indices"], f["X/indptr"]), shape=(f["X/indptr"].shape[0]-1, 19331))
    mat.indices = mat.indices.astype("int16")
    f_out.append(mat, "central")
    f.close()

f_out.close()



f_out = h5torch.File("../cellxgene.h5t", "a")

obs_ = np.stack([obs[i].cat.codes.values for i in obs.columns[2:]]).T
categories = {str(i)+"_"+k : obs[k].cat.categories.values.astype(bytes) for i, k in enumerate(obs.columns[2:])}

f_out.register(obs_, axis = 0, name = "obs", dtype_load="int64")

for k, v in categories.items():
    f_out.register(v, axis = "unstructured", name = k, dtype_save="bytes", dtype_load="str")

def read_var(path):
    return anndata.read_h5ad(path, backed='r').var

var = read_var(files[0])
f_out.register(var.values.astype(bytes), axis = 1, name = "var", dtype_save="bytes", dtype_load="str")
f_out.close()


import h5torch
split_h5 = np.full(len(obs), "NA", dtype=object)
split_h5[splits2["train"]] = "train"
split_h5[splits2["val"]] = "val"
split_h5[splits2["test"]] = "test"

f_out = h5torch.File("../cellxgene.h5t", "a")
f_out.register(split_h5.astype(bytes), axis = 0, name = "split", dtype_save="bytes", dtype_load="str")
f_out.close()