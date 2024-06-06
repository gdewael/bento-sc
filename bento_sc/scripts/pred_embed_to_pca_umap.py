import sys
input_file_raw = str(sys.argv[1])
input_file_pca = str(sys.argv[2])
output_file_pca = str(sys.argv[3]) 
output_file_umap = str(sys.argv[4])
steps = str(sys.argv[5]) #"pca", "umap", "both"

from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples
import umap
import numpy as np

def ASWct(embeddings, celltypes):
    k = silhouette_samples(embeddings, celltypes)
    return np.mean((k + 1)/2)

def ASWbatch(embeddings, batch):
    k = silhouette_samples(embeddings, batch)
    return np.mean(1 - np.abs(k))

def raw_to_pca(input_file_raw, output_file_pca):
    inputs = np.load(input_file_raw)
    obs = inputs["obs"]
    embeds = inputs["embeds"]

    embeds_pca = PCA(n_components=64).fit_transform(embeds)
    np.savez(output_file_pca, obs=obs, embeds=embeds_pca)
    print(ASWct(embeds_pca, obs[:, 3]))
    print(ASWbatch(embeds_pca, obs[:, 0]))
    return obs, embeds_pca

def pca_to_umap(input_file_pca, output_file_umap):
    inputs = np.load(input_file_pca)
    obs = inputs["obs"]
    embeds = inputs["embeds"]
    reducer = umap.UMAP(verbose=True, min_dist=0.5)
    embeds_umap = reducer.fit_transform(embeds)
    np.savez(output_file_umap, obs=obs, embeds=embeds_umap)
    return obs, embeds_umap


if steps == "pca":
    _ = raw_to_pca(input_file_raw, output_file_pca)
if steps == "umap":
    _ = pca_to_umap(input_file_pca, output_file_umap)
if steps == "both":
    _ = raw_to_pca(input_file_raw, output_file_pca)
    _ = pca_to_umap(input_file_pca, output_file_umap)