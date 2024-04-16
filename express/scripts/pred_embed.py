
import torch
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples
from sklearn.preprocessing import StandardScaler
from express.data import ExpressDataModule
import sys
from tqdm import tqdm

config_path = str(sys.argv[1])
baseline = str(sys.argv[2])
n_threads = str(sys.argv[3])
embeddings_save_path = str(sys.argv[4])

import os

os.environ["NUMBA_NUM_THREADS"] = n_threads

dm = ExpressDataModule(
    config_path
)
dm.setup(None)

embeds = []
obs = []
for batch in tqdm(dm.test_dataloader()):
    embeds.append(batch["gene_counts"])
    obs.append(batch["0/obs"])

embeds = torch.cat(embeds).numpy()
obs = torch.cat(obs).numpy()

ss = StandardScaler()
pca = PCA(n_components=64)
embeds = ss.fit_transform(embeds)
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