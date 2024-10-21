import h5torch
from bento_sc.data import CellSampleProcessor, FilterTopGenes, SequentialPreprocessor
import numpy as np
from tqdm import tqdm
import sys
import obonet
import networkx
import h5torch

url = "https://github.com/obophenotype/cell-ontology/releases/download/v2023-05-22/cl-simple.obo"
graph = obonet.read_obo(url, ignore_obsolete=True)

# only use "is_a" edges
edges_to_delete = []
for i, x in enumerate(graph.edges):
    if x[2] != "is_a":
        edges_to_delete.append((x[0], x[1]))
for x in edges_to_delete:
    graph.remove_edge(u=x[0], v=x[1])

# define mapping from id to name
id_to_name = {id_: data.get("name") for id_, data in graph.nodes(data=True)}
# define inverse mapping from name to id
name_to_id = {v: k for k, v in id_to_name.items()}

def find_child_nodes(cell_type):
    return [
        id_to_name[node] for node in networkx.ancestors(graph, name_to_id[cell_type])
    ]

def find_parent_nodes(cell_type):
    return [
        id_to_name[node] for node in networkx.descendants(graph, name_to_id[cell_type])
    ]

myeloid_cells = ["myeloid cell"] + find_child_nodes("myeloid cell")
B_cells = ["B cell"] + find_child_nodes("B cell")
NK_cells = ["natural killer cell"] + find_child_nodes("natural killer cell")
T_cells = ["T cell"] + find_child_nodes("T cell")

f_cxg = h5torch.File("../data/cellxgene.h5t")
ct_cxg = f_cxg["unstructured/3_cell_type"][:].astype(str)

celltypes_indices = []
for celltype_to_select in [myeloid_cells, B_cells, NK_cells, T_cells]:
    indices_celltype = np.where(np.isin(ct_cxg, np.array([ct for ct in celltype_to_select if ct in ct_cxg])))[0]
    celltypes_indices.append(np.isin(f_cxg["0/obs"][:, 3], indices_celltype))

blood_cells = (f_cxg["0/obs"][:, 7] == np.where(f_cxg["unstructured"]["7_tissue_general"][:] == b"blood")[0][0])
celltype_indices_blood = []
for celltype_ind in celltypes_indices:
    celltype_indices_blood.append(np.logical_and(celltype_ind, blood_cells))


val_or_test = str(sys.argv[1])

frac = f_cxg["0/split"][:] == bytes(val_or_test, 'utf-8')
celltype_indices_blood_frac = []
for celltype_ind in celltype_indices_blood:
    celltype_indices_blood_frac.append(np.logical_and(celltype_ind, frac))


from scipy.sparse import csr_matrix

matrix = np.zeros((10_000, 19331), dtype="int32")
obs_ = []
subsets = []

c = 0
for ct_ind in celltype_indices_blood_frac:
    d = h5torch.Dataset(
        "../data/cellxgene.h5t",
        sample_processor=CellSampleProcessor(
            SequentialPreprocessor(
                FilterTopGenes(affected_keys=["gene_counts", "gene_index", "gene_counts_true"], number=1024)
            ), return_zeros=False
        ),
        subset=np.where(ct_ind)[0])


    subset = np.random.choice(len(d), (2500,), replace=False)
    subsets.append(subset)
    for n in tqdm(subset):
        matrix[c, d[n]["gene_index"].numpy()] = d[n]["gene_counts"].numpy().astype(np.int32)
        c+=1
        obs_.append(d[n]["0/obs"])


celltype = np.array(
    ["Myeloid cells"] * 2500 +
    ["B cells"] * 2500 +
    ["NK cells"] * 2500 +
    ["T cells"] * 2500
).astype(bytes)

obs = np.stack(obs_)

f_out = h5torch.File("../data/cellxgene_grn_%s.h5t" % val_or_test, "w")
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
    celltype,
    axis=0,
    name="celltype",
    dtype_save="bytes",
    dtype_load="str"
)

f_out.register(
    d.f["1/var"][:],
    axis=1,
    name="var",
    dtype_save="bytes",
    dtype_load="str",
)

f_out.register(
    np.concatenate(subsets),
    axis="unstructured",
    name="samples",
)

split = np.full((10_000), "test")
f_out.register(
    split,
    axis=0,
    name="split",
    dtype_save="bytes",
    dtype_load="str"
)

f_out.close()