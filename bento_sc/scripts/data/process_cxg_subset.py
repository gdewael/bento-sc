import h5torch
from bento_sc.data import CellSampleProcessor, FilterTopGenes, SequentialPreprocessor
import numpy as np
from tqdm import tqdm
import sys


val_or_test = str(sys.argv[1])

d = h5torch.Dataset(
    "../data/cellxgene.h5t",
    sample_processor=CellSampleProcessor(
        SequentialPreprocessor(), return_zeros=False
    ),
    subset=("0/split", val_or_test))


seqdepths = []
for strt, stp in tqdm(zip(
    d.f["central/indptr"][:][d.indices],
    d.f["central/indptr"][:][d.indices+1]
), total=len(d.indices)):
    seqdepths.append(d.f["central/data"][strt:stp].sum())

k = np.random.choice(
    np.where(np.array(seqdepths)>25_000)[0],
    size=(25_000, ),
    replace=False,
)
matrix = np.zeros((25_000, 19331), dtype="int32")
from scipy.sparse import csr_matrix
obs_ = []
for ix, n in tqdm(enumerate(k)):
    matrix[ix, d[n]["gene_index"].numpy()] = d[n]["gene_counts"].numpy().astype(np.int32)
    obs_.append(d[n]["0/obs"])


obs = np.stack(obs_)
f_out = h5torch.File("../data/cellxgene_sub_%s.h5t" % val_or_test, "w")
f_out.register(
    csr_matrix(matrix),
    axis="central",
    mode="csr",
    dtype_save="float32",
    dtype_load="float32",
    csr_load_sparse=True
)
f_out.register(
    obs,
    axis=0,
    name="obs",
    dtype_save="int64",
    dtype_load="int64"
)

f_out.register(
    d.f["1/var"][:],
    axis=1,
    name="var",
    dtype_save="bytes",
    dtype_load="str",
)

f_out.register(
    k,
    axis="unstructured",
    name="samples",
)

split = np.full((25_000), "test")
f_out.register(
    split,
    axis=0,
    name="split",
    dtype_save="bytes",
    dtype_load="str"
)

f_out.close()