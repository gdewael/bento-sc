from os.path import join
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import obonet
import networkx
from importlib.resources import files
import cellxgene_census
import argparse
import os
import anndata
from math import ceil
import h5py
import sys
from scipy.sparse import csr_matrix
import urllib.request
import gzip
import shutil
import h5torch


class CustomFormatter(
    argparse.ArgumentDefaultsHelpFormatter, argparse.MetavarTypeHelpFormatter
):
    pass


def main():
    parser = argparse.ArgumentParser(
        description="Data downloading script launching pad. Choose a datafile to download and process to h5torch format.",
        formatter_class=CustomFormatter,
    )

    parser.add_argument(
        "datafile",
        type=str,
        metavar="datafile",
        choices=["scTab", "perturb", "neurips_citeseq", "replogle_perturb"],
        help="Datafile to download, choices: {%(choices)s}",
    )

    args = parser.parse_args(sys.argv[1:2])

    mapper = {
        "scTab": (get_sctab_parser, get_sctab),
        "neurips_citeseq": (get_citeseq_parser, get_citeseq),
        "replogle_perturb": (get_perturb_parser, get_perturb),
        "batchcorr_embryolimb": (get_el_parser, get_el),
        "batchcorr_greatapes": (get_ga_parser, get_ga),
        "batchcorr_circimm": (get_ci_parser, get_ci),
    }

    parser_func, download_func = mapper[args.datafile]
    parser = parser_func()
    args = parser.parse_args(sys.argv[2:])
    download_func(args)


def get_sctab_parser():
    parser = argparse.ArgumentParser(
        description="Download and format scTab.",
        formatter_class=CustomFormatter,
    )
    parser.add_argument(
        "base_folder",
        type=str,
        metavar="base_folder",
        help="base folder to put intermediate files in (will occupy +-256gb). Note that this script does not delete intermediate files.",
    )
    parser.add_argument(
        "output_file_path",
        type=str,
        metavar="output_file_path",
        help="output .h5t file to write processed data to.",
    )
    return parser


def get_citeseq_parser():
    parser = argparse.ArgumentParser(
        description="Download and format NeurIPS 2021 CITE-seq dataset.",
        formatter_class=CustomFormatter,
    )
    parser.add_argument(
        "base_folder",
        type=str,
        metavar="base_folder",
        help="base folder to put intermediate files in. Note that this script does not delete intermediate files.",
    )
    parser.add_argument(
        "sctab_h5t",
        type=str,
        metavar="sctab_h5t",
        help="path to processed sctab h5t file. Used to unify gene vocabulary.",
    )
    parser.add_argument(
        "output_file_path",
        type=str,
        metavar="output_file_path",
        help="output .h5t file to write processed data to.",
    )
    return parser


def get_perturb_parser():
    parser = argparse.ArgumentParser(
        description="Download and format Replogle Perturb-seq dataset.",
        formatter_class=CustomFormatter,
    )
    parser.add_argument(
        "base_folder",
        type=str,
        metavar="base_folder",
        help="base folder to put intermediate files in. Note that this script does not delete intermediate files.",
    )
    parser.add_argument(
        "sctab_h5t",
        type=str,
        metavar="sctab_h5t",
        help="path to processed sctab h5t file. Used to unify gene vocabulary.",
    )
    parser.add_argument(
        "output_file_path",
        type=str,
        metavar="output_file_path",
        help="output .h5t file to write processed data to.",
    )
    return parser


def get_el_parser():
    parser = argparse.ArgumentParser(
        description="Download and format Embryonic Limb dataset.",
        formatter_class=CustomFormatter,
    )
    parser.add_argument(
        "base_folder",
        type=str,
        metavar="base_folder",
        help="base folder to put intermediate files in. Note that this script does not delete intermediate files.",
    )
    parser.add_argument(
        "sctab_h5t",
        type=str,
        metavar="sctab_h5t",
        help="path to processed sctab h5t file. Used to unify gene vocabulary.",
    )
    parser.add_argument(
        "output_file_path",
        type=str,
        metavar="output_file_path",
        help="output .h5t file to write processed data to.",
    )
    return parser


def get_ga_parser():
    parser = argparse.ArgumentParser(
        description="Download and Great Apes dataset.",
        formatter_class=CustomFormatter,
    )
    parser.add_argument(
        "base_folder",
        type=str,
        metavar="base_folder",
        help="base folder to put intermediate files in. Note that this script does not delete intermediate files.",
    )
    parser.add_argument(
        "sctab_h5t",
        type=str,
        metavar="sctab_h5t",
        help="path to processed sctab h5t file. Used to unify gene vocabulary.",
    )
    parser.add_argument(
        "output_file_path",
        type=str,
        metavar="output_file_path",
        help="output .h5t file to write processed data to.",
    )
    return parser


def get_ci_parser():
    parser = argparse.ArgumentParser(
        description="Download and format Circulating Immune cells dataset.",
        formatter_class=CustomFormatter,
    )
    parser.add_argument(
        "base_folder",
        type=str,
        metavar="base_folder",
        help="base folder to put intermediate files in. Note that this script does not delete intermediate files.",
    )
    parser.add_argument(
        "sctab_h5t",
        type=str,
        metavar="sctab_h5t",
        help="path to processed sctab h5t file. Used to unify gene vocabulary.",
    )
    parser.add_argument(
        "output_file_path",
        type=str,
        metavar="output_file_path",
        help="output .h5t file to write processed data to.",
    )
    return parser


def get_sctab(args):
    # in large part adapted from https://github.com/theislab/scTab,
    # with some changes to make it work for limited RAM systems.
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
            id_to_name[node]
            for node in networkx.ancestors(graph, name_to_id[cell_type])
        ]

    def find_parent_nodes(cell_type):
        return [
            id_to_name[node]
            for node in networkx.descendants(graph, name_to_id[cell_type])
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
        pd.DataFrame(
            {"n_children": n_children, "n_parents": n_parents}, index=cell_types
        )
        .query("n_parents <= 7")
        .index.tolist()
    )
    cell_types_to_remove = list(set(cell_types_to_remove))

    obs_subset = obs[~obs.cell_type.isin(cell_types_to_remove)].copy()
    for col in obs_subset.columns:
        if obs_subset[col].dtype == "category":
            obs_subset[col] = obs_subset[col].cat.remove_unused_categories()

    protein_coding_genes = pd.read_parquet(
        "https://raw.githubusercontent.com/theislab/scTab/devel/notebooks/store_creation/features.parquet"
    ).gene_names.tolist()

    # download in batches to not run out of memory
    for i, idxs in tqdm(
        enumerate(np.array_split(obs_subset.soma_joinid.to_numpy(), 100))
    ):
        adata = cellxgene_census.get_anndata(
            census=census,
            organism="Homo sapiens",
            X_name="raw",
            obs_coords=idxs.tolist(),
            var_value_filter=f"feature_name in {protein_coding_genes}",
            column_names={"obs": COLUMN_NAMES, "var": ["feature_id", "feature_name"]},
        )
        adata.write_h5ad(join(args.base_folder, f"{i}.h5ad"))

    files = [
        join(args.base_folder, file)
        for file in sorted(
            os.listdir(args.base_folder), key=lambda x: int(x.split(".")[0])
        )
        if file.endswith(".h5ad")
    ]

    def read_obs(path):
        obs = anndata.read_h5ad(path, backed="r").obs
        obs["tech_sample"] = obs.dataset_id.astype(str) + "_" + obs.donor_id.astype(str)
        return obs

    # read obs
    print("Loading obs...")
    obs = pd.concat([read_obs(file) for file in files]).reset_index(drop=True)
    for col in obs.columns:
        if obs[col].dtype == object:
            obs[col] = obs[col].astype("category")
            obs[col].cat.remove_unused_categories()

    def get_split(samples, val_split: float = 0.15, test_split: float = 0.15, seed=1):
        rng = np.random.default_rng(seed=seed)

        samples = np.array(samples)
        rng.shuffle(samples)
        n_samples = len(samples)

        n_samples_val = ceil(val_split * n_samples)
        n_samples_test = ceil(test_split * n_samples)
        n_samples_train = n_samples - n_samples_val - n_samples_test

        return {
            "train": samples[:n_samples_train],
            "val": samples[n_samples_train : (n_samples_train + n_samples_val)],
            "test": samples[(n_samples_train + n_samples_val) :],
        }

    def subset(splits, frac):
        assert 0.0 < frac <= 1.0
        if frac == 1.0:
            return splits
        else:
            return splits[: ceil(frac * len(splits))]

    splits = {"train": [], "val": [], "test": []}
    tech_sample_splits = get_split(obs.tech_sample.unique().tolist())
    for x in ["train", "val", "test"]:
        # tech_samples are already shuffled in the get_split method -> just subselect to subsample donors
        if x == "train":
            # only subset training data set
            splits[x] = obs[
                obs.tech_sample.isin(subset(tech_sample_splits[x], 1.0))
            ].index.to_numpy()
        else:
            splits[x] = obs[
                obs.tech_sample.isin(tech_sample_splits[x])
            ].index.to_numpy()

    assert len(np.intersect1d(splits["train"], splits["val"])) == 0
    assert len(np.intersect1d(splits["train"], splits["test"])) == 0
    assert len(np.intersect1d(splits["val"], splits["train"])) == 0
    assert len(np.intersect1d(splits["val"], splits["test"])) == 0

    rng = np.random.default_rng(seed=1)

    splits["train"] = rng.permutation(splits["train"])
    splits["val"] = rng.permutation(splits["val"])
    splits["test"] = rng.permutation(splits["test"])

    splits2 = {}
    splits2["train"] = splits["train"][: (len(splits["train"]) // 1024) * 1024]
    splits2["val"] = splits["val"][: (len(splits["val"]) // 1024) * 1024]
    splits2["test"] = splits["test"][: (len(splits["test"]) // 1024) * 1024]

    len_ = len(obs)

    f_out = h5torch.File(args.output_file_path, "w")

    f = h5py.File(os.path.join(args.base_folder, "0.h5ad"))
    mat = csr_matrix(
        (f["X/data"], f["X/indices"], f["X/indptr"].astype(np.int64)),
        shape=(f["X/indptr"].shape[0] - 1, 19331),
    )
    mat.indices = mat.indices.astype("int16")
    mat.indptr = mat.indptr.astype("int64")
    f_out.register(
        mat,
        axis="central",
        mode="csr",
        dtype_save="float32",
        dtype_load="float32",
        csr_load_sparse=True,
        length=len_,
    )
    f.close()

    for i in tqdm(range(1, 100)):
        f = h5py.File(os.path.join(args.base_folder, "%s.h5ad" % i))
        mat = csr_matrix(
            (f["X/data"], f["X/indices"], f["X/indptr"]),
            shape=(f["X/indptr"].shape[0] - 1, 19331),
        )
        mat.indices = mat.indices.astype("int16")
        mat.indptr = mat.indptr.astype("int64")
        f_out.append(mat, "central")
        f.close()

    f_out.close()

    f_out = h5torch.File(args.output_file_path, "a")

    obs_ = np.stack([obs[i].cat.codes.values for i in obs.columns[2:]]).T
    categories = {
        str(i) + "_" + k: obs[k].cat.categories.values.astype(bytes)
        for i, k in enumerate(obs.columns[2:])
    }

    f_out.register(obs_, axis=0, name="obs", dtype_load="int64")

    for k, v in categories.items():
        f_out.register(
            v, axis="unstructured", name=k, dtype_save="bytes", dtype_load="str"
        )

    def read_var(path):
        return anndata.read_h5ad(path, backed="r").var

    var = read_var(files[0])
    f_out.register(
        var.values.astype(bytes),
        axis=1,
        name="var",
        dtype_save="bytes",
        dtype_load="str",
    )
    f_out.close()

    import h5torch

    split_h5 = np.full(len(obs), "NA", dtype=object)
    split_h5[splits2["train"]] = "train"
    split_h5[splits2["val"]] = "val"
    split_h5[splits2["test"]] = "test"

    val_sub = np.random.permutation(splits2["val"])[:100_000]
    split_h5[val_sub] = "val_sub"

    f_out = h5torch.File(args.output_file_path, "a")
    f_out.register(
        split_h5.astype(bytes),
        axis=0,
        name="split",
        dtype_save="bytes",
        dtype_load="str",
    )
    f_out.close()
    return None


def get_perturb(args):
    urllib.request.urlretrieve(
        "https://plus.figshare.com/ndownloader/files/35773219",
        os.path.join(args.base_folder, "./perturb.h5ad"),
    )

    f = h5py.File(os.path.join(args.base_folder, "./perturb.h5ad"))

    gene_ids_pert = f["var/gene_id"][:]
    f_cxg = h5py.File(args.sctab_h5t)
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

    keep_sample_indices = np.array(
        [ix for ix, i in enumerate(sample_filter[:-1]) if ~np.isnan(i)]
        + [len(sample_filter) - 1]
    )
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

    splits = np.load(files("bento_sc.utils.data").joinpath("split_pert.npz"))
    split = splits["split"]
    matched_control = splits["matched_control"]
    train_control_indices = splits["train_control_indices"]

    f = h5torch.File(args.output_file_path, "w")
    f.register(
        csr_matrix(new_matrix),
        axis="central",
        mode="csr",
        dtype_save="float32",
        dtype_load="float32",
        csr_load_sparse=True,
    )

    f.register(
        train_control_indices,
        axis="unstructured",
        name="train_control_indices",
        dtype_save="int64",
        dtype_load="int64",
    )

    f.register(
        matched_control,
        axis=0,
        name="matched_control",
        dtype_save="int64",
        dtype_load="int64",
    )

    f.register(split, axis=0, name="split", dtype_save="bytes", dtype_load="str")

    f.register(
        new_pert_array,
        axis=0,
        name="perturbed_gene",
        dtype_save="float64",
        dtype_load="float64",
    )

    f.register(
        f_cxg["1/var"][:],
        axis=1,
        name="var",
        dtype_save="bytes",
        dtype_load="str",
    )

    f.close()
    return None


def get_citeseq(args):
    urllib.request.urlretrieve(
        "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE194nnn/GSE194122/suppl/GSE194122%5Fopenproblems%5Fneurips2021%5Fcite%5FBMMC%5Fprocessed%2Eh5ad%2Egz",
        os.path.join(args.base_folder, "./citeseq.h5ad.gz"),
    )

    with gzip.open(os.path.join(args.base_folder, "./citeseq.h5ad.gz"), "rb") as f_in:
        with open(os.path.join(args.base_folder, "./citeseq.h5ad"), "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)

    f = h5py.File(os.path.join(args.base_folder, "./citeseq.h5ad"))
    f.visititems(print)

    central = f["layers/counts"]
    matrix = csr_matrix(
        (central["data"][:], central["indices"][:], central["indptr"][:])
    ).toarray()

    GEX_columns = (
        f["var/__categories/feature_types"][:][f["var/feature_types"][:]] == b"GEX"
    )
    ADT_columns = (
        f["var/__categories/feature_types"][:][f["var/feature_types"][:]] == b"ADT"
    )

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

    f_cxg = h5py.File(args.sctab_h5t)
    gene_ids_cxg = f_cxg["1/var"][:, 0]

    indices_map = []
    for g_id in gene_ids_gex:
        match = np.where(gene_ids_cxg == g_id)[0]
        if len(match) > 0:
            indices_map.append(match[0])
        else:
            indices_map.append(np.nan)
    indices_map = np.array(indices_map)

    new_matrix = np.zeros((matrix.shape[0], 19331), dtype="int32")
    for ix, l in tqdm(enumerate(indices_map)):
        if ~np.isnan(l):
            new_matrix[:, l.astype(int)] = GEX_matrix[:, ix]

    f = h5torch.File(args.output_file_path, "w")

    f.register(
        csr_matrix(new_matrix),
        axis="central",
        mode="csr",
        dtype_save="float32",
        dtype_load="float32",
        csr_load_sparse=True,
    )

    f.register(
        split.astype(bytes), axis=0, name="split", dtype_save="bytes", dtype_load="str"
    )

    f.register(
        ADT_matrix, axis=0, name="ADT", dtype_save="float32", dtype_load="float32"
    )

    obs = np.stack([donor_id, site, celltypes]).T
    f.register(obs, axis=0, name="obs", dtype_save="int64", dtype_load="int64")

    f.register(
        np.array(["donor_id", "site", "celltypes"]).astype(bytes),
        axis="unstructured",
        name="obs_columbs",
        dtype_save="bytes",
        dtype_load="str",
    )

    f.register(
        celltype_cats.astype(bytes),
        axis="unstructured",
        name="celltype_names",
        dtype_save="bytes",
        dtype_load="str",
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
        dtype_load="str",
    )

    f.close()
    return None


def get_el(args):
    urllib.request.urlretrieve(
        "https://datasets.cellxgene.cziscience.com/6d0d9779-f66b-4297-9bbd-4f232651fb99.h5ad",
        os.path.join(args.base_folder, "./embryonic_limb.h5ad"),
    )

    f = h5py.File(os.path.join(args.base_folder, "./embryonic_limb.h5ad"))

    assay_cats = f["obs/assay/categories"][:]
    assay_codes = f["obs/assay/codes"][:]

    cell_type_cats = f["obs/cell_type/categories"][:]
    cell_type_codes = f["obs/cell_type/codes"][:]

    donor_cats = f["obs/donor_id/categories"][:]
    donor_codes = f["obs/donor_id/codes"][:]

    matrix = csr_matrix(
        (f["raw/X/data"][:], f["raw/X/indices"][:], f["raw/X/indptr"][:])
    ).toarray()

    gene_ids = f["var/feature_name/categories"][:][f["var/feature_name/codes"][:]]

    f_cxg = h5py.File(args.sctab_h5t)
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

    f = h5torch.File(args.output_file_path, "w")

    f.register(
        csr_matrix(new_matrix),
        axis="central",
        mode="csr",
        dtype_save="float32",
        dtype_load="float32",
        csr_load_sparse=True,
    )

    obs = np.stack([assay_codes, donor_codes, cell_type_codes]).T
    f.register(obs, axis=0, name="obs", dtype_save="int64", dtype_load="int64")

    f.register(
        assay_cats.astype(bytes),
        axis="unstructured",
        name="0_assay_categories",
        dtype_save="bytes",
        dtype_load="str",
    )

    f.register(
        donor_cats.astype(bytes),
        axis="unstructured",
        name="1_donor_categories",
        dtype_save="bytes",
        dtype_load="str",
    )

    f.register(
        cell_type_cats.astype(bytes),
        axis="unstructured",
        name="3_cell_type_categories",
        dtype_save="bytes",
        dtype_load="str",
    )

    split = np.full((len(obs)), "test")
    f.register(split, axis=0, name="split", dtype_save="bytes", dtype_load="str")

    f.register(
        f_cxg["1/var"][:],
        axis=1,
        name="var",
        dtype_save="bytes",
        dtype_load="str",
    )

    f.close()
    return None


def get_ga(args):
    urllib.request.urlretrieve(
        "https://datasets.cellxgene.cziscience.com/b9103b10-e021-4c20-9714-60b32928179e.h5ad",
        os.path.join(args.base_folder, "./great_apes.h5ad"),
    )

    f = h5py.File(os.path.join(args.base_folder, "./great_apes.h5ad"))

    assay_cats = f["obs/assay/categories"][:]
    assay_codes = f["obs/assay/codes"][:]

    cell_type_cats = f["obs/cell_type/categories"][:]
    cell_type_codes = f["obs/cell_type/codes"][:]

    dev_cats = f["obs/development_stage/categories"][:]
    dev_codes = f["obs/development_stage/codes"][:]

    donor_cats = f["obs/donor_id/categories"][:]
    donor_codes = f["obs/donor_id/codes"][:]

    matrix = csr_matrix(
        (f["raw/X/data"][:], f["raw/X/indices"][:], f["raw/X/indptr"][:])
    ).toarray()

    gene_ids = f["var/gene"][:]

    f_cxg = h5py.File(args.sctab_h5t)
    gene_ids_cxg = f_cxg["1/var"][:, 0]

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

    f = h5torch.File(args.output_file_path, "w")

    f.register(
        csr_matrix(new_matrix),
        axis="central",
        mode="csr",
        dtype_save="float32",
        dtype_load="float32",
        csr_load_sparse=True,
    )

    obs = np.stack([assay_codes, dev_codes, donor_codes, cell_type_codes]).T
    f.register(obs, axis=0, name="obs", dtype_save="int64", dtype_load="int64")

    f.register(
        assay_cats.astype(bytes),
        axis="unstructured",
        name="0_assay_categories",
        dtype_save="bytes",
        dtype_load="str",
    )

    f.register(
        dev_cats.astype(bytes),
        axis="unstructured",
        name="1_development_stage_categories",
        dtype_save="bytes",
        dtype_load="str",
    )

    f.register(
        donor_cats.astype(bytes),
        axis="unstructured",
        name="2_donor_categories",
        dtype_save="bytes",
        dtype_load="str",
    )

    f.register(
        cell_type_cats.astype(bytes),
        axis="unstructured",
        name="3_cell_type_categories",
        dtype_save="bytes",
        dtype_load="str",
    )

    split = np.full((len(obs)), "test")
    f.register(split, axis=0, name="split", dtype_save="bytes", dtype_load="str")

    f.register(
        f_cxg["1/var"][:],
        axis=1,
        name="var",
        dtype_save="bytes",
        dtype_load="str",
    )

    f.close()
    return None


def get_ci(args):
    urllib.request.urlretrieve(
        "https://datasets.cellxgene.cziscience.com/e1d1d48b-f0ac-41e6-b7d7-283f709b6421.h5ad",
        os.path.join(args.base_folder, "./circ_imm.h5ad"),
    )

    f = h5py.File(os.path.join(args.base_folder, "./circ_imm.h5ad"))

    assay_cats = f["obs/assay/categories"][:]
    assay_codes = f["obs/assay/codes"][:]

    cell_type_cats = f["obs/cell_type/categories"][:]
    cell_type_codes = f["obs/cell_type/codes"][:]

    donor_cats = f["obs/donor_id/categories"][:]
    donor_codes = f["obs/donor_id/codes"][:]

    matrix = csr_matrix(
        (f["raw/X/data"][:], f["raw/X/indices"][:], f["raw/X/indptr"][:])
    ).toarray()

    gene_ids = f["var/feature_name/categories"][:][f["var/feature_name/codes"][:]]

    f_cxg = h5py.File(args.sctab_h5t)
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

    f = h5torch.File(args.output_file_path, "w")

    f.register(
        csr_matrix(new_matrix),
        axis="central",
        mode="csr",
        dtype_save="float32",
        dtype_load="float32",
        csr_load_sparse=True,
    )

    obs = np.stack([assay_codes, donor_codes, cell_type_codes]).T
    f.register(obs, axis=0, name="obs", dtype_save="int64", dtype_load="int64")

    f.register(
        assay_cats.astype(bytes),
        axis="unstructured",
        name="0_assay_categories",
        dtype_save="bytes",
        dtype_load="str",
    )

    f.register(
        donor_cats.astype(bytes),
        axis="unstructured",
        name="1_donor_categories",
        dtype_save="bytes",
        dtype_load="str",
    )

    f.register(
        cell_type_cats.astype(bytes),
        axis="unstructured",
        name="3_cell_type_categories",
        dtype_save="bytes",
        dtype_load="str",
    )

    split = np.full((len(obs)), "test")
    f.register(split, axis=0, name="split", dtype_save="bytes", dtype_load="str")

    f.register(
        f_cxg["1/var"][:],
        axis=1,
        name="var",
        dtype_save="bytes",
        dtype_load="str",
    )

    f.close()
    return None


if __name__ == "__main__":
    main()
