import os
os.environ["OMP_NUM_THREADS"] = "4" # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = "4" # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = "4" # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = "4" # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = "4" # export NUMEXPR_NUM_THREADS=1

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:2048"


import sys
input_h5t = str(sys.argv[1])
input_model_embeds = str(sys.argv[2])
output_h5ad = str(sys.argv[3])
ct_col = str(sys.argv[4])
batch_col = str(sys.argv[5])
skip_pca = str(sys.argv[6])

from sklearn.decomposition import PCA
import numpy as np
from scipy.sparse import csr_matrix
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import h5torch
import anndata as ad
import bbknn
import scib
import scanpy as sc

f = h5torch.File(input_h5t)
f = f.to_dict()

matrix = csr_matrix((f["central/data"][:],f["central/indices"][:],f["central/indptr"][:]), shape = (f["0/obs"].shape[0], f["1/var"].shape[0]))

adata = ad.AnnData(matrix)
adata.obs = pd.DataFrame(f["0/obs"], columns=np.arange(f["0/obs"].shape[1]).astype(str))
adata.var = pd.DataFrame(f["1/var"], columns=np.arange(f["1/var"].shape[1]).astype(str))

adata.obs[batch_col] = adata.obs[batch_col].astype("category")
adata.obs[ct_col] = adata.obs[ct_col].astype("category")

file = np.load(input_model_embeds)
adata.obsm["X_emb"] = file["embeds"]

if skip_pca == "True":
    embeds_pca = adata.obsm["X_emb"]
else:
    embeds_pca = PCA(n_components=50).fit_transform(adata.obsm["X_emb"])

adata.obsm["X_pca"] = embeds_pca

bbknn.bbknn(adata, batch_key=batch_col)

sc.tl.umap(adata)

clisi, ilisi = scib.me.lisi_graph(adata, batch_key=batch_col, label_key=ct_col, type_="knn", n_cores=16)
graph_conn = scib.me.graph_connectivity(adata, label_key=ct_col)

scib.cl.cluster_optimal_resolution(adata, cluster_key="iso_label", label_key=ct_col)
iso_f1 = scib.me.isolated_labels_f1(adata, batch_key=batch_col, label_key=ct_col, embed=None)
ari = scib.me.ari(adata, cluster_key="iso_label", label_key=ct_col)
nmi = scib.me.nmi(adata, cluster_key="iso_label", label_key=ct_col)

adata.uns["scores"] = {
    "iLISI" : ilisi,
    "Graph Connectivity" : graph_conn,
    "cLISI" : clisi,
    "ARI" : ari,
    "NMI" : nmi,
    "Isolated F1": iso_f1,
}

adata.write(output_h5ad)