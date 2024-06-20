import h5torch
import h5py
from scipy.sparse import csr_matrix
import numpy as np
from tqdm import tqdm

# "https://datasets.cellxgene.cziscience.com/d6e81c75-6af5-4ed5-b5d1-e99e49c0ba00"
INPUT_FILE = "/data/home/gaetandw/express/data/embryonic_limb.h5ad"
OUTPUT_FILE = "/data/home/gaetandw/express/data/embryonic_limb.h5t"
CXG_FILE = "/data/home/gaetandw/express/data/cellxgene.h5t"

f = h5py.File(INPUT_FILE)

assay_cats = f["obs/assay/categories"][:]
assay_codes = f["obs/assay/codes"][:]

cell_type_cats = f["obs/cell_type/categories"][:]
cell_type_codes = f["obs/cell_type/codes"][:]

donor_cats = f["obs/donor_id/categories"][:]
donor_codes = f["obs/donor_id/codes"][:]

matrix = csr_matrix((f["raw/X/data"][:], f["raw/X/indices"][:], f["raw/X/indptr"][:])).toarray()

gene_ids = f["var/feature_name/categories"][:][f["var/feature_name/codes"][:]]

f_cxg = h5py.File(CXG_FILE)
gene_ids_cxg = f_cxg["1/var"][:, 1]


indices_map = []
for g_id in gene_ids:
    match = np.where(gene_ids_cxg == g_id)[0]
    if len(match) > 0:
        indices_map.append(match[0])
    else:
        indices_map.append(np.nan)
indices_map = np.array(indices_map)

new_matrix = np.zeros((matrix.shape[0], 19331), dtype="int32")
for ix, l in tqdm(enumerate(indices_map)):
    if ~np.isnan(l):
        new_matrix[:, l.astype(int)] = matrix[:, ix]

f = h5torch.File(OUTPUT_FILE, "w")

f.register(
    csr_matrix(new_matrix),
    axis="central",
    mode="csr",
    dtype_save="float32",
    dtype_load="float32",
    csr_load_sparse=True
)

obs = np.stack([assay_codes, donor_codes, cell_type_codes]).T
f.register(
    obs,
    axis=0,
    name="obs",
    dtype_save="int64",
    dtype_load="int64"
)

f.register(
    assay_cats.astype(bytes),
    axis="unstructured",
    name="0_assay_categories",
    dtype_save="bytes",
    dtype_load="str"
)

f.register(
    donor_cats.astype(bytes),
    axis="unstructured",
    name="1_donor_categories",
    dtype_save="bytes",
    dtype_load="str"
)

f.register(
    cell_type_cats.astype(bytes),
    axis="unstructured",
    name="3_cell_type_categories",
    dtype_save="bytes",
    dtype_load="str"
)

split = np.full((len(obs)), "test")
f.register(
    split,
    axis=0,
    name="split",
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

f.close()