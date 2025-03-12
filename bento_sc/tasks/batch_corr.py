import os

os.environ["OMP_NUM_THREADS"] = "4"  # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = "4"  # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = "4"  # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"  # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = "4"  # export NUMEXPR_NUM_THREADS=1
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:2048"
import torch
import numpy as np
from bento_sc.data import BentoDataModule
from bento_sc.models import BentoTransformer
from bento_sc.utils.config import Config
import argparse
from scipy.sparse import csr_matrix
import pandas as pd
from sklearn.decomposition import PCA
import h5torch
import anndata as ad
import bbknn
import scib
import scanpy as sc
import sys


class CustomFormatter(
    argparse.ArgumentDefaultsHelpFormatter, argparse.MetavarTypeHelpFormatter
):
    pass


def main():
    parser = argparse.ArgumentParser(
        description="Script for batch correction evaluation.",
        formatter_class=CustomFormatter,
    )

    parser.add_argument(
        "embed_or_correct",
        type=str,
        metavar="embed_or_correct",
        choices=["embed", "correct"],
        help="This script contains two subroutines: (1) creating embeddings from a pre-trained model, and (2) performing batch correction and computing scIB scores.",
    )

    args = parser.parse_args(sys.argv[1:2])

    if args.embed_or_correct == "embed":
        embed_parser = get_embed_parser()
        args = embed_parser.parse_args(sys.argv[2:])
        embed(args)

    if args.embed_or_correct == "correct":
        correct_parser = get_correct_parser()
        args = correct_parser.parse_args(sys.argv[2:])
        correct(args)


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
    return parser


def get_correct_parser():
    parser = argparse.ArgumentParser(
        description="Perform batch correction and compute scIB scores.",
        formatter_class=CustomFormatter,
    )
    parser.add_argument(
        "h5t_data_file", type=str, metavar="h5t_data_file", help="h5torch data file"
    )
    parser.add_argument(
        "input_model_embeds",
        type=str,
        metavar="input_model_embeds",
        help="save_path from previous step",
    )
    parser.add_argument(
        "output_h5ad",
        type=str,
        metavar="output_h5ad",
        help="results will be written in a h5ad file",
    )
    parser.add_argument(
        "batch_col",
        type=str,
        metavar="batch_col",
        help="index of column in 0/obs in h5t file where batch effects are",
    )
    parser.add_argument(
        "ct_col",
        type=str,
        metavar="ct_col",
        help="index of column in 0/obs in h5t file where celltypes are",
    )
    return parser


def embed(args):
    config = Config(args.config_path)

    if args.data_path is not None:
        config["data_path"] = args.data_path

    dm = BentoDataModule(config)
    dm.setup(None)

    model = BentoTransformer.load_from_checkpoint(args.checkpoint)

    device_ = "cuda:%s" % config.devices[0]
    model = model.to(device_).to(torch.bfloat16).eval()

    obs = []
    embeds = []
    with torch.no_grad():
        for batch in dm.predict_dataloader():
            batch["gene_index"] = batch["gene_index"].to(model.device)
            batch["gene_counts"] = batch["gene_counts"].to(model.device)
            batch["gene_counts_true"] = batch["gene_counts_true"].to(model.device)

            if not model.config.discrete_input:
                batch["gene_counts"] = batch["gene_counts"].to(model.dtype)
            else:
                batch["gene_counts"] = batch["gene_counts"].float()

            y = model(batch)

            embeds.append(y[:, 0].cpu())
            obs.append(batch["0/obs"])

    embeds = torch.cat(embeds).float().numpy()
    obs = torch.cat(obs).numpy()

    np.savez(args.save_path, obs=obs, embeds=embeds)
    return None


def correct(args):
    f = h5torch.File(args.h5t_data_file)
    f = f.to_dict()

    matrix = csr_matrix(
        (f["central/data"][:], f["central/indices"][:], f["central/indptr"][:]),
        shape=(f["0/obs"].shape[0], f["1/var"].shape[0]),
    )

    adata = ad.AnnData(matrix)
    adata.obs = pd.DataFrame(
        f["0/obs"], columns=np.arange(f["0/obs"].shape[1]).astype(str)
    )
    adata.var = pd.DataFrame(
        f["1/var"], columns=np.arange(f["1/var"].shape[1]).astype(str)
    )

    adata.obs[args.batch_col] = adata.obs[args.batch_col].astype("category")
    adata.obs[args.ct_col] = adata.obs[args.ct_col].astype("category")

    file = np.load(args.input_model_embeds)
    adata.obsm["X_emb"] = file["embeds"]

    embeds_pca = PCA(n_components=50).fit_transform(adata.obsm["X_emb"])

    adata.obsm["X_pca"] = embeds_pca

    bbknn.bbknn(adata, batch_key=args.batch_col)

    sc.tl.umap(adata)

    clisi, ilisi = scib.me.lisi_graph(
        adata, batch_key=args.batch_col, label_key=args.ct_col, type_="knn", n_cores=16
    )
    graph_conn = scib.me.graph_connectivity(adata, label_key=args.ct_col)

    scib.cl.cluster_optimal_resolution(
        adata, cluster_key="iso_label", label_key=args.ct_col
    )
    iso_f1 = scib.me.isolated_labels_f1(
        adata, batch_key=args.batch_col, label_key=args.ct_col, embed=None
    )
    ari = scib.me.ari(adata, cluster_key="iso_label", label_key=args.ct_col)
    nmi = scib.me.nmi(adata, cluster_key="iso_label", label_key=args.ct_col)

    adata.uns["scores"] = {
        "iLISI": ilisi,
        "Graph Connectivity": graph_conn,
        "cLISI": clisi,
        "ARI": ari,
        "NMI": nmi,
        "Isolated F1": iso_f1,
    }
    adata.write(args.output_h5ad)
    return None


if __name__ == "__main__":
    main()
