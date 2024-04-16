from os.path import join
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import obonet
import networkx
import cellxgene_census
import sys

BASE_PATH = str(sys.argv[1])

# url = 'http://purl.obolibrary.org/obo/cl/cl-simple.obo'
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


census = cellxgene_census.open_soma(census_version="2023-05-15")

summary = census["census_info"]["summary"]

PROTOCOLS = [
    "10x 5' v2",
    "10x 3' v3",
    "10x 3' v2",
    "10x 5' v1",
    "10x 3' v1",
    "10x 3' transcription profiling",
    "10x 5' transcription profiling",
]

COLUMN_NAMES = [
    "soma_joinid",
    "is_primary_data",
    "dataset_id",
    "donor_id",
    "assay",
    "cell_type",
    "development_stage",
    "disease",
    "tissue",
    "tissue_general",
]

obs = (
    census["census_data"]["homo_sapiens"]
    .obs.read(
        column_names=COLUMN_NAMES,
        value_filter=f"is_primary_data == True and assay in {PROTOCOLS}",
    )
    .concat()
    .to_pandas()
)

obs["tech_sample"] = (obs.dataset_id + "_" + obs.donor_id).astype("category")

for col in COLUMN_NAMES:
    if obs[col].dtype == object:
        obs[col] = obs[col].astype("category")

# remove all cell types which are not a subtype of native cell
cell_types_to_remove = (
    obs[~obs.cell_type.isin(find_child_nodes("native cell"))]
    .cell_type.unique()
    .tolist()
)

# remove all cell types which have less than 5000 cells
cell_freq = obs.cell_type.value_counts()
cell_types_to_remove += cell_freq[cell_freq < 5000].index.tolist()

# remove cell types which have less than 30 tech_samples
tech_samples_per_cell_type = (
    obs[["cell_type", "tech_sample"]]
    .groupby("cell_type")
    .agg({"tech_sample": "nunique"})
    .sort_values("tech_sample")
)
cell_types_to_remove += tech_samples_per_cell_type[
    tech_samples_per_cell_type.tech_sample <= 30
].index.tolist()

# filter out too granular labels
# remove all cells that have <= 7 parents in the cell ontology
cell_types = obs.cell_type.unique().tolist()

n_children = []
n_parents = []

for cell_type in cell_types:
    n_parents.append(len(find_parent_nodes(cell_type)))
    n_children.append(len(find_child_nodes(cell_type)))

cell_types_to_remove += (
    pd.DataFrame({"n_children": n_children, "n_parents": n_parents}, index=cell_types)
    .query("n_parents <= 7")
    .index.tolist()
)
cell_types_to_remove = list(set(cell_types_to_remove))

obs_subset = obs[~obs.cell_type.isin(cell_types_to_remove)].copy()
for col in obs_subset.columns:
    if obs_subset[col].dtype == "category":
        obs_subset[col] = obs_subset[col].cat.remove_unused_categories()

cell_types_to_keep = obs_subset.cell_type.unique().tolist()

protein_coding_genes = pd.read_parquet(
    "https://raw.githubusercontent.com/theislab/scTab/devel/notebooks/store_creation/features.parquet"
).gene_names.tolist()


# download in batches to not run out of memory
for i, idxs in tqdm(enumerate(np.array_split(obs_subset.soma_joinid.to_numpy(), 100))):
    adata = cellxgene_census.get_anndata(
        census=census,
        organism="Homo sapiens",
        X_name="raw",
        obs_coords=idxs.tolist(),
        var_value_filter=f"feature_name in {protein_coding_genes}",
        column_names={"obs": COLUMN_NAMES, "var": ["feature_id", "feature_name"]},
    )
    adata.write_h5ad(join(BASE_PATH, f"{i}.h5ad"))
