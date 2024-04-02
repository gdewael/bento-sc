import h5py
from scipy.sparse import csr_matrix
import numpy as np
import pandas as pd
from tqdm import tqdm
import h5torch

f = h5py.File("../GSE194122_openproblems_neurips2021_cite_BMMC_processed.h5ad")
f.visititems(print)

central = f["layers/counts"]
matrix = csr_matrix((central["data"][:], central["indices"][:], central["indptr"][:])).toarray()

GEX_columns = f["var/__categories/feature_types"][:][f["var/feature_types"][:]] == b"GEX"
ADT_columns = f["var/__categories/feature_types"][:][f["var/feature_types"][:]] == b"ADT"

GEX_matrix = matrix[:, np.where(GEX_columns)[0]]
ADT_matrix = matrix[:, np.where(ADT_columns)[0]]

donor_id = f["obs/DonorID"][:]
site = f["obs/Site"][:]
split = f["obs/__categories/is_train"][:][f["obs/is_train"][:]]
split[split == b"iid_holdout"] = b"val"
celltypes = f["obs/cell_type"][:]
celltype_cats = f["obs/__categories/cell_type"][:]

gene_ids = f["var/__categories/gene_id"][:][f["var/gene_id"][:]]
gene_ids_gex = gene_ids[GEX_columns]
gene_ids_adt = gene_ids[ADT_columns]

f_cxg = h5py.File("../data/cellxgene.h5t")
gene_ids_cxg = f_cxg["1/var"][:, 0]

indices_map = []
for g_id in gene_ids_gex:
    match = np.where(gene_ids_cxg == g_id)[0]
    if len(match) > 0:
        indices_map.append(match[0])
    else:
        indices_map.append(np.nan)
indices_map = np.array(indices_map)

new_matrix = np.zeros((matrix.shape[0], 19331), dtype="int16")
for ix, l in tqdm(enumerate(indices_map)):
    if ~np.isnan(l):
        new_matrix[:, l.astype(int)] = GEX_matrix[:, ix]


f = h5torch.File("../citeseq.h5t", "w")

f.register(
    csr_matrix(new_matrix),
    axis="central",
    mode="csr",
    dtype_save="float32",
    dtype_load="float32",
    csr_load_sparse=True
)

f.register(
    split.astype(bytes),
    axis=0,
    name="split",
    dtype_save="bytes",
    dtype_load="str"
)

f.register(
    ADT_matrix,
    axis=0,
    name="ADT",
    dtype_save="float32",
    dtype_load="float32"
)

obs = np.stack([donor_id, site, celltypes]).T
f.register(
    obs,
    axis=0,
    name="obs",
    dtype_save="int64",
    dtype_load="int64"
)

f.register(
    np.array(["donor_id", "site", "celltypes"]).astype(bytes),
    axis="unstructured",
    name="obs_columbs",
    dtype_save="bytes",
    dtype_load="str"
)

f.register(
    celltype_cats.astype(bytes),
    axis="unstructured",
    name="celltype_names",
    dtype_save="bytes",
    dtype_load="str"
)

f.register(
    f_cxg["1/var"][:],
    axis=1,
    name="var",
    dtype_save="bytes",
    dtype_load="str",
)

f.register(
    gene_ids_adt.astype(bytes),
    axis="unstructured",
    name="gene_ids_adt",
    dtype_save="bytes",
    dtype_load="str"
)

f.close()