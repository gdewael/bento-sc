import h5torch
from scipy.sparse import csr_matrix
import h5py
import numpy as np
from tqdm import tqdm

f = h5py.File("../data_unprocessed/perturb.h5ad")

gene_ids_pert = f["var/gene_id"][:]
f_cxg = h5py.File("../data/cellxgene.h5t")
gene_ids_cxg = f_cxg["1/var"][:, 0]

indices_map = []
for g_id in gene_ids_pert:
    match = np.where(gene_ids_cxg == g_id)[0]
    if len(match) > 0:
        indices_map.append(match[0])
    else:
        indices_map.append(np.nan)
indices_map = np.array(indices_map)

gene_ids_pert_samples = f["obs/__categories/gene_id"][:]

sample_filter = []
for g_id in gene_ids_pert_samples:
    if (g_id not in gene_ids_pert) and (g_id != b"non-targeting"):
        sample_filter.append(np.nan)
    elif (g_id not in gene_ids_cxg) and (g_id != b"non-targeting"):
        sample_filter.append(np.nan)
    elif g_id != b"non-targeting":
        match = np.where(gene_ids_cxg == g_id)[0]
        sample_filter.append(match[0])
    else:
        sample_filter.append("control")
sample_filter = np.array(sample_filter, dtype="object")


matrix = f["X"][:].astype("int32")
new_matrix = np.zeros((310385, 19331), dtype="int32")
for ix, l in tqdm(enumerate(indices_map)):
    if ~np.isnan(l):
        new_matrix[:, l.astype(int)] = matrix[:, ix]

keep_sample_indices = np.array([ix for ix, i in enumerate(sample_filter[:-1]) if ~np.isnan(i)] + [len(sample_filter)-1])
gene_id_indices = f["obs/gene_id"][:]
new_matrix = new_matrix[np.isin(gene_id_indices, keep_sample_indices)]

new_pert_array = []
for ff in f["obs/gene_id"][:]:
    if sample_filter[ff] == "control":
        new_pert_array.append(np.nan)
    elif np.isnan(sample_filter[ff]):
        continue
    else:
        new_pert_array.append(sample_filter[ff])

new_pert_array = np.array(new_pert_array, allow_pickle=True)

splits = np.load("../bento-sc/bento_sc/utils/data/split_pert.npz")
split = splits["split"]
matched_control = splits["matched_control"]
train_control_indices = splits["train_control_indices"]

f = h5torch.File("../perturb.h5t", "w")
f.register(
    csr_matrix(new_matrix),
    axis="central",
    mode="csr",
    dtype_save="float32",
    dtype_load="float32",
    csr_load_sparse=True
)

f.register(
    train_control_indices,
    axis="unstructured",
    name="train_control_indices",
    dtype_save="int64",
    dtype_load="int64"
)

f.register(
    matched_control,
    axis=0,
    name="matched_control",
    dtype_save="int64",
    dtype_load="int64"
)

f.register(
    split,
    axis=0,
    name="split",
    dtype_save="bytes",
    dtype_load="str"
)

f.register(
    new_pert_array,
    axis=0,
    name="perturbed_gene",
    dtype_save="float64",
    dtype_load="float64"
)

f.register(
    f_cxg["1/var"][:],
    axis=1,
    name="var",
    dtype_save="bytes",
    dtype_load="str",
)

f.close()