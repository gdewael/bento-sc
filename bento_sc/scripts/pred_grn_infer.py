from bento_sc.utils.config import Config
from bento_sc.data import *
from bento_sc.models import BentoTransformer
from bento_sc.utils.metrics import pearson_batch_masked
from lightning.pytorch.plugins.environments import LightningEnvironment
from lightning.pytorch import Trainer
import torch
import numpy as np
import h5torch
from importlib.resources import files
from tqdm import tqdm
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import argparse
import h5py


def boolean(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")

def main():
    class CustomFormatter(
        argparse.ArgumentDefaultsHelpFormatter, argparse.MetavarTypeHelpFormatter
    ):
        pass

    parser = argparse.ArgumentParser(
        description="Training script for reconstruction.",
        formatter_class=CustomFormatter,
    )

    parser.add_argument("config_path", type=str, metavar="config_path", help="config_path")
    parser.add_argument("approach", type=str, metavar="approach", help="approach (model)")
    parser.add_argument("pertdata_path", type=str, help="Path to external validation perturbation data")
    parser.add_argument("--data_path", type=str, default=None, help="Data file. Overrides value in config file if specified")
    parser.add_argument("--test_mode", type=str, default="val", help="val or test")
    parser.add_argument("--counts_as_pos", type=boolean, default=False)

    args = parser.parse_args()

    config = Config(args.config_path)
    if args.data_path is not None:
        config["data_path"] = args.data_path
    
    dm = BentoDataModule(
        config
    )
    dm.setup(None)

    model = BentoTransformer.load_from_checkpoint(args.approach)

    model = model.eval().to(config.devices[0]).to(torch.bfloat16)

    embeddings_per_gene = {
        "Myeloid cells" : {i : [] for i in range(19331)},
        "B cells" : {i : [] for i in range(19331)},
        "NK cells" : {i : [] for i in range(19331)},
        "T cells" : {i : [] for i in range(19331)},
    }

    key = ("gene_index" if not args.counts_as_pos else "gene_counts")

    with torch.no_grad():
        for batch in dm.test_dataloader():
            batch["gene_counts"] = batch["gene_counts"].to(model.dtype).to(model.device)
            batch["gene_index"] = batch["gene_index"].to(model.device)
            y = model(batch).cpu()

            for sample in range(len(batch["gene_index"])):
                for gene in range(batch["gene_index"].shape[1]):
                    if batch["gene_counts"][sample, gene] != -1:
                        embeddings_per_gene[batch["0/celltype"][sample]][
                            batch[key][sample, gene].item()
                        ].append(y[:, 1:][sample, gene].cpu())

    gene_lists_celltype = {}
    top_k_celltype = {}
    for celltype in embeddings_per_gene:
        aggregated_per_cell = {k : torch.stack(v).mean(0) for k, v in embeddings_per_gene[celltype].items() if len(v) > 10}

        embeddings = torch.zeros((19331, 512))

        for k1, v1 in aggregated_per_cell.items():
            embeddings[k1] = v1

        embeddings_norm = embeddings / (embeddings.norm(dim=1)[:, None]+1e-8)

        A = torch.einsum("k h, q h -> k q", embeddings_norm, embeddings_norm)

        A_reduced = A[torch.where(A.sum(1))[0]][:, torch.where(A.sum(1))[0]]

        A_reduced = A_reduced - torch.diag(torch.ones(len(A_reduced)))

        top_k = torch.argsort(-(A_reduced).abs(), dim = 1)
        gene_list = torch.where(A.sum(1))[0]
        top_k = gene_list[top_k]

        gene_lists_celltype[celltype] = gene_list
        top_k_celltype[celltype] = top_k



    f = h5py.File(args.pertdata_path)
    gene_ids_pert = f["var"]["_index"][:]
    gene_ids_cxg = dm.train.f["1/var"][:, 1]

    pert_to_cxg_indices = []
    for g_id in gene_ids_pert:
        match = np.where(gene_ids_cxg == g_id)[0]
        if len(match) > 0:
            pert_to_cxg_indices.append(match[0])
        else:
            pert_to_cxg_indices.append(np.nan)
    pert_to_cxg_indices = np.array(pert_to_cxg_indices)

    cxg_to_pert_indices = []
    for g_id in gene_ids_cxg:
        match = np.where(gene_ids_pert == g_id)[0]
        if len(match) > 0:
            cxg_to_pert_indices.append(match[0])
        else:
            cxg_to_pert_indices.append(np.nan)
    cxg_to_pert_indices = np.array(cxg_to_pert_indices)


    path = files("bento_sc.utils.data").joinpath("allTFs_hg38.txt")
    TFs = np.loadtxt(path, dtype="str").astype(bytes)
    TFs_in_pert_indices = []
    for ix, g_id in enumerate(gene_ids_pert):
        match = np.where(TFs == g_id)[0]
        if len(match) > 0:
            TFs_in_pert_indices.append(ix)
        else:
            TFs_in_pert_indices.append(np.nan)
    TFs_in_pert_indices = np.array(TFs_in_pert_indices)

    layer = f["layers"]["scgen_pearson"][:]

    score_per_celltype = []
    for celltype in gene_lists_celltype:
        index_of_celltype = np.where(f["obs"]["cell_type"]["categories"][:].astype(str) == celltype)[0][0]
        indices_celltype_in_pertdata = f["obs"]["cell_type"]["codes"][:] == index_of_celltype
        dataset = layer[indices_celltype_in_pertdata]
        groups = f["obs/sm_name/codes"][:][indices_celltype_in_pertdata]

        splits = [te for _, te in GroupKFold(n_splits=10).split(dataset, None, groups)]

        gene_list_ct = gene_lists_celltype[celltype]
        top_k_ct = top_k_celltype[celltype]

        top_k_ct_filtered = top_k_ct[~np.isnan(cxg_to_pert_indices[gene_list_ct])]
        gene_list_ct_filtered = gene_list_ct[~np.isnan(cxg_to_pert_indices[gene_list_ct])]

        gene_list_ct_filtered_pert = cxg_to_pert_indices[gene_list_ct_filtered]
        top_k_ct_filtered_pert = cxg_to_pert_indices[top_k_ct_filtered]

        hvg_pert_indices = np.argsort(np.var(dataset, 0))[::-1]
        to_train_on = hvg_pert_indices[np.where(np.isin(hvg_pert_indices, gene_list_ct_filtered_pert))[0][:500]]

        gene_list_to_train_on = gene_list_ct_filtered_pert[np.isin(gene_list_ct_filtered_pert, to_train_on)]
        top_k_list_to_train_on = top_k_ct_filtered_pert[np.isin(gene_list_ct_filtered_pert, to_train_on)]

        scores_per_gene = []
        
        for g, top_k_g in tqdm(zip(
            gene_list_to_train_on,
            top_k_list_to_train_on
        ), total=len(gene_list_to_train_on)):
            
            top_features_in_pert = top_k_g[~np.isnan(top_k_g)].astype(int)
            top_features_in_pert_and_TF = TFs_in_pert_indices[top_features_in_pert]
            top_features_in_pert_and_TF = top_features_in_pert_and_TF[~np.isnan(top_features_in_pert_and_TF)].astype(int)

            selected_features = top_features_in_pert_and_TF[:5]

            preds_gene_per_cv = []
            trues_gene_per_cv = []
            for i in np.arange(-1, len(splits)-1):

                if args.test_mode == "val":
                    val = i
                    test = i+1
                    train_indices = np.concatenate([s for s, i in zip(splits, ~np.isin(np.arange(len(splits)), [val, test])) if i])
                    test_indices = splits[val]

                elif args.test_mode == "test":
                    test = i+1
                    train_indices = np.concatenate([s for s, i in zip(splits, ~np.isin(np.arange(len(splits)), [test])) if i])
                    test_indices = splits[test]

                y_train = dataset[train_indices][:, g.astype(int)]
                y_test = dataset[test_indices][:, g.astype(int)]

                X_train = dataset[train_indices][:, selected_features.astype(int)]
                X_test = dataset[test_indices][:, selected_features.astype(int)]


                linreg = LinearRegression().fit(X_train, y_train)

                preds_gene_per_cv.append(linreg.predict(X_test))
                trues_gene_per_cv.append(y_test)

            r2_ = r2_score(np.concatenate(trues_gene_per_cv), np.concatenate(preds_gene_per_cv))
            scores_per_gene.append(r2_)

        print(
            np.mean(scores_per_gene),
            np.min(scores_per_gene),
            np.max(scores_per_gene),
            np.median(scores_per_gene)
        )

        score_per_celltype.append(np.mean(scores_per_gene))
            
    print(np.mean(score_per_celltype))


if __name__ == "__main__":
    main()