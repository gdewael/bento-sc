import sys
config_path = str(sys.argv[1])
baseline = str(sys.argv[2])
n_threads = str(sys.argv[3])
embeddings_save_path = str(sys.argv[4])

import os
os.environ["NUMBA_NUM_THREADS"] = n_threads
os.environ["OMP_NUM_THREADS"] = n_threads
os.environ["OPENBLAS_NUM_THREADS"] = n_threads
os.environ["MKL_NUM_THREADS"] = n_threads
os.environ["VECLIB_MAXIMUM_THREADS"] = n_threads
os.environ["NUMEXPR_NUM_THREADS"] = n_threads

import torch
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples
from bento_sc.data import BentoDataModule
from bento_sc.utils.config import Config

from tqdm import tqdm

config = Config(config_path)

dm = BentoDataModule(
    config
)
dm.setup(None)

embeds = []
obs = []
for batch in tqdm(dm.test_dataloader()):
    embeds.append(batch["gene_counts"])
    obs.append(batch["0/obs"])

embeds = torch.cat(embeds).numpy()
obs = torch.cat(obs).numpy()

#ss = StandardScaler()
pca = PCA(n_components=64)
#embeds = ss.fit_transform(embeds)
embeds = pca.fit_transform(embeds)

def ASWct(embeddings, celltypes):
    k = silhouette_samples(embeddings, celltypes)
    return np.mean((k + 1)/2)

def ASWbatch(embeddings, batch):
    k = silhouette_samples(embeddings, batch)
    return np.mean(1 - np.abs(k))

print(ASWct(embeds, obs[:, 3]))
print(ASWbatch(embeds, obs[:, 0]))
np.save(embeddings_save_path, embeds)