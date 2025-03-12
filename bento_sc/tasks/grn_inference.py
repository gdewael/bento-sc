from bento_sc.utils.config import Config
from bento_sc.data import *
from bento_sc.models import BentoTransformer
import torch
import pandas as pd
import numpy as np
from importlib.resources import files
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
import argparse
import h5py
import os
import sys


def boolean(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


class CustomFormatter(
    argparse.ArgumentDefaultsHelpFormatter, argparse.MetavarTypeHelpFormatter
):
    pass


def main():
    parser = argparse.ArgumentParser(
        description="Script for GRN inference evaluation.",
        formatter_class=CustomFormatter,
    )

    parser.add_argument(
        "embed_or_eval",
        type=str,
        metavar="embed_or_eval",
        choices=["embed", "eval"],
        help="This script contains two subroutines: (1) creating embeddings from a pre-trained model, and (2) inferring a GRN and evaluating it through a regression proxy task.",
    )

    args = parser.parse_args(sys.argv[1:2])

    if args.embed_or_eval == "embed":
        embed_parser = get_embed_parser()
        args = embed_parser.parse_args(sys.argv[2:])
        embed(args)

    if args.embed_or_eval == "eval":
        eval_parser = get_eval_parser()
        args = eval_parser.parse_args(sys.argv[2:])
        eval_grn(args)


def get_embed_parser():
    parser = argparse.ArgumentParser(
        description="Create embeddings from a pre-trained model.",
        formatter_class=CustomFormatter,
    )
    parser.add_argument(
        "config_path", type=str, metavar="config_path", help="config_path"
    )
    parser.add_argument(
        "checkpoint",
        type=str,
        metavar="checkpoint",
        help="pre-trained model checkpoint",
    )
    parser.add_argument("save_path", type=str, metavar="save_path", help="save_path")
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Data file. Overrides value in config file if specified",
    )
    parser.add_argument(
        "--counts_as_pos",
        type=boolean,
        default=False,
        help="Flag to enable if your pre-trained models treated counts as positional indices.",
    )

    return parser


def get_eval_parser():
    parser = argparse.ArgumentParser(
        description="Infer a GRN from embeddings and evaluate.",
        formatter_class=CustomFormatter,
    )
    parser.add_argument(
        "config_path", type=str, metavar="config_path", help="config_path"
    )
    parser.add_argument(
        "path_to_embeddings",
        type=str,
        metavar="path_to_embeddings",
        help="path_to_embeddings",
    )
    parser.add_argument(
        "pertdata_path", type=str, help="Path to external validation perturbation data"
    )
    parser.add_argument(
        "scenic_database", type=str, help="SCENIC motif to gene binding database"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Data file. Overrides value in config file if specified",
    )
    parser.add_argument(
        "--test_mode",
        type=str,
        default="val",
        choices=["val", "test"],
        help="val or test",
    )
    parser.add_argument(
        "--n_features",
        type=int,
        default=5,
        help="max number of features to use in the regression task",
    )
    parser.add_argument(
        "--motifs_per_gene",
        type=int,
        default=500,
        help="max number of motifs per gene to allow",
    )
    parser.add_argument(
        "--n_genes_to_train_on",
        type=int,
        default=200,
        help="number of regression models to train.",
    )
    return parser


def embed(args):
    config = Config(args.config_path)
    if args.data_path is not None:
        config["data_path"] = args.data_path

    dm = BentoDataModule(config)
    dm.setup(None)

    model = BentoTransformer.load_from_checkpoint(args.checkpoint)

    model = model.eval().to(config.devices[0]).to(torch.bfloat16)

    embeddings_per_gene = {
        "Myeloid cells": {i: [] for i in range(19331)},
        "B cells": {i: [] for i in range(19331)},
        "NK cells": {i: [] for i in range(19331)},
        "T cells": {i: [] for i in range(19331)},
    }

    key = "gene_index" if not args.counts_as_pos else "gene_counts"

    with torch.no_grad():
        for batch in tqdm(dm.test_dataloader(), total=len(dm.test_dataloader())):
            if not model.config.discrete_input:
                batch["gene_counts"] = (
                    batch["gene_counts"].to(model.dtype).to(model.device)
                )
            else:
                batch["gene_counts"] = batch["gene_counts"].float().to(model.device)

            batch["gene_index"] = batch["gene_index"].to(model.device)
            y = model(batch).cpu()

            for sample in range(len(batch["gene_index"])):
                for gene in range(batch["gene_index"].shape[1]):
                    if batch["gene_counts"][sample, gene] != -1:
                        embeddings_per_gene[batch["0/celltype"][sample]][
                            batch[key][sample, gene].item()
                        ].append(y[:, 1:][sample, gene].cpu())

    embeddings_per_celltype = {}
    for celltype in embeddings_per_gene:
        aggregated_per_cell = {
            k: torch.stack(v).mean(0)
            for k, v in embeddings_per_gene[celltype].items()
            if len(v) > 10
        }

        embeddings = torch.zeros((19331, 512))

        for k1, v1 in aggregated_per_cell.items():
            embeddings[k1] = v1

        embeddings_per_celltype[celltype] = embeddings

    torch.save(
        embeddings_per_celltype["NK cells"],
        os.path.join(args.save_path, "embeddings_NK.pt"),
    )
    torch.save(
        embeddings_per_celltype["T cells"],
        os.path.join(args.save_path, "embeddings_T.pt"),
    )
    torch.save(
        embeddings_per_celltype["Myeloid cells"],
        os.path.join(args.save_path, "embeddings_Myeloid.pt"),
    )
    torch.save(
        embeddings_per_celltype["B cells"],
        os.path.join(args.save_path, "embeddings_B.pt"),
    )
    return None


def eval_grn(args):
    config = Config(args.config_path)
    if args.data_path is not None:
        config["data_path"] = args.data_path

    dm = BentoDataModule(config)
    dm.setup(None)

    embeddings_per_celltype = {
        "NK cells": torch.load(
            os.path.join(args.path_to_embeddings, "embeddings_NK.pt")
        ),
        "T cells": torch.load(os.path.join(args.path_to_embeddings, "embeddings_T.pt")),
        "B cells": torch.load(os.path.join(args.path_to_embeddings, "embeddings_B.pt")),
        "Myeloid cells": torch.load(
            os.path.join(args.path_to_embeddings, "embeddings_Myeloid.pt")
        ),
    }

    gene_lists_celltype = {}
    top_k_celltype = {}
    gene_ids_cxg = dm.train.f["1/var"][:, 1]
    for celltype in embeddings_per_celltype:
        embeddings = embeddings_per_celltype[celltype]

        embeddings_norm = embeddings / (embeddings.norm(dim=1)[:, None] + 1e-8)

        A = torch.einsum("k h, q h -> k q", embeddings_norm, embeddings_norm)

        A_reduced = A[torch.where(A.sum(1))[0]][:, torch.where(A.sum(1))[0]]

        A_reduced = A_reduced - torch.diag(torch.ones(len(A_reduced)))

        top_k = torch.argsort(-(A_reduced).abs(), dim=1)
        gene_list = torch.where(A.sum(1))[0]
        top_k = gene_list[top_k]

        gene_lists_celltype[celltype] = gene_ids_cxg[gene_list.numpy()]
        top_k_celltype[celltype] = gene_ids_cxg[top_k.numpy()]

    motif2gene_db = pd.read_feather(args.scenic_database)
    motif2tf_db = pd.read_table(
        files("bento_sc.utils.data").joinpath(
            "motifs-v10nr_clust-nr.hgnc-m0.001-o0.0.tbl"
        )
    )

    path = files("bento_sc.utils.data").joinpath("allTFs_hg38.txt")
    TFs = np.loadtxt(path, dtype="str").astype(bytes)

    TFs_in_motif2tf = motif2tf_db["gene_name"].values.astype(bytes)

    motif2tf_db = motif2tf_db.iloc[np.isin(TFs_in_motif2tf, TFs)]

    motif2tf_dict = {}
    for m, tf in zip(motif2tf_db["#motif_id"], motif2tf_db["gene_name"]):
        if m not in motif2tf_dict:
            motif2tf_dict[m] = [tf]
        else:
            if tf not in motif2tf_dict[m]:
                motif2tf_dict[m].append(tf)

    normalized = motif2gene_db.iloc[:, :-1].values / (
        motif2gene_db.iloc[:, :-1].values.mean(1)[:, None] + 1e-8
    )
    ranking = np.argsort(normalized, axis=0)[::-1][: args.motifs_per_gene]
    gene_2_tf_dict = {}
    for gene, rank in zip(motif2gene_db.columns[:-1], ranking.T):
        motifs_for_gene = motif2gene_db.iloc[:, -1].values[rank]
        tfs_for_gene = []
        for motif in motifs_for_gene:
            if motif in motif2tf_dict:
                tfs_for_gene += motif2tf_dict[motif]
        gene_2_tf_dict[gene] = list(np.unique(tfs_for_gene))

    gene_2_tf_dict_filt = {k: v for k, v in gene_2_tf_dict.items()}

    f = h5py.File(args.pertdata_path)
    gene_ids_pert = f["var"]["_index"][:]
    gene_ids_scenic = np.array(list(gene_2_tf_dict_filt)).astype(bytes)
    tf_ids_scenic = np.array(
        list((set([t for v in gene_2_tf_dict_filt.values() for t in v])))
    ).astype(bytes)

    possible_gene_ids = np.array(
        list(set(list(gene_ids_pert)).intersection(gene_ids_scenic))
    )

    possible_TF_ids = np.array(
        list(set(list(gene_ids_pert)).intersection(tf_ids_scenic))
    )

    gene_lists_celltype_filt = {}
    top_k_celltype_filt = {}
    for ct in gene_lists_celltype.keys():
        gene_list_ct = gene_lists_celltype[ct]
        top_k_ct = top_k_celltype[ct]

        index_select = np.isin(gene_list_ct, possible_gene_ids)
        gene_lists_celltype_filt[ct] = gene_list_ct[index_select]
        top_k_celltype_filt[ct] = top_k_ct[index_select]

    dataset = f["layers"]["scgen_pearson"][:]

    dataset_f_select = np.isin(gene_ids_pert, possible_gene_ids)
    dataset_genes = dataset[:, dataset_f_select]
    dataset_genes_f = gene_ids_pert[dataset_f_select]

    dataset_f_select = np.isin(gene_ids_pert, possible_TF_ids)
    dataset_tf = dataset[:, dataset_f_select]
    dataset_tf_f = gene_ids_pert[dataset_f_select]

    score_per_celltype = []
    for celltype in gene_lists_celltype:
        index_of_celltype = np.where(
            f["obs"]["cell_type"]["categories"][:].astype(str) == celltype
        )[0][0]
        indices_celltype_in_pertdata = (
            f["obs"]["cell_type"]["codes"][:] == index_of_celltype
        )
        dataset_genes_celltype = dataset_genes[indices_celltype_in_pertdata]
        dataset_tf_celltype = dataset_tf[indices_celltype_in_pertdata]

        groups = f["obs/sm_name/codes"][:][indices_celltype_in_pertdata]
        train_groups, test_groups = np.split(
            np.unique(groups), [int(len(np.unique(groups)) * 0.8)]
        )
        # get the first 1_000 highly variable genes in the pert data which are in the
        # celltype-spec GRN
        hvg_ = dataset_genes_f[np.argsort(np.var(dataset_genes_celltype, 0))[::-1]]
        to_train_on = hvg_[
            np.where(np.isin(hvg_, gene_lists_celltype_filt[celltype]))[0][
                : args.n_genes_to_train_on
            ]
        ]

        score = []
        for i in tqdm(
            range((0 if args.test_mode == "val" else 1), args.n_genes_to_train_on, 2),
            total=args.n_genes_to_train_on // 2,
        ):
            selected_gene = to_train_on[i]

            index_of_gene = np.where(
                gene_lists_celltype_filt[celltype] == selected_gene
            )[0]

            top_sim_genes_ranked = top_k_celltype_filt[celltype][index_of_gene[0]]

            tfs_for_that_gene = np.array(
                gene_2_tf_dict[selected_gene.decode("utf-8")]
            ).astype(bytes)

            top_sim_tfs_ranked = top_sim_genes_ranked[
                np.isin(top_sim_genes_ranked, tfs_for_that_gene)
            ]

            top_sim_tfs_in_pert = top_sim_tfs_ranked[
                np.isin(top_sim_tfs_ranked, dataset_tf_f)
            ][: args.n_features]

            if len(top_sim_tfs_in_pert) == 0:
                score.append(0)
            else:
                top_sim_tfs_in_pert = top_sim_tfs_in_pert
                Y = dataset_genes_celltype[
                    :, np.where(dataset_genes_f == selected_gene)[0]
                ]
                X = dataset_tf_celltype[:, np.isin(dataset_tf_f, top_sim_tfs_in_pert)]

                X_train = X[np.isin(groups, train_groups)]
                X_test = X[np.isin(groups, test_groups)]
                Y_train = Y[np.isin(groups, train_groups)]
                Y_test = Y[np.isin(groups, test_groups)]

                linreg = LinearRegression().fit(X_train, Y_train)
                score_ = linreg.score(X_test, Y_test)
                if score_ < 0:
                    score.append(0)
                else:
                    score.append(score_)
        score_per_celltype.append(np.mean(score))

    print(args.path_to_embeddings)
    print(np.mean(score_per_celltype))
    return None


if __name__ == "__main__":
    main()
