from bento_sc.utils.config import Config
from bento_sc.data import *
from bento_sc.models import BentoTransformer
from bento_sc.utils.metrics import pearson_batch_masked
from lightning.pytorch.plugins.environments import LightningEnvironment
from lightning.pytorch import Trainer
import torch
import pandas as pd
import numpy as np
from importlib.resources import files
from tqdm import tqdm
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import argparse
import h5py
import os

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
    parser.add_argument("save_path", type=str, metavar="save_path", help="save_path")
    parser.add_argument("--data_path", type=str, default=None, help="Data file. Overrides value in config file if specified")
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
        for batch in tqdm(dm.test_dataloader(), total=len(dm.test_dataloader())):
            if not model.config.discrete_input:
                batch["gene_counts"] = batch["gene_counts"].to(model.dtype).to(model.device)
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
        aggregated_per_cell = {k : torch.stack(v).mean(0) for k, v in embeddings_per_gene[celltype].items() if len(v) > 10}

        embeddings = torch.zeros((19331, 512))

        for k1, v1 in aggregated_per_cell.items():
            embeddings[k1] = v1
        
        embeddings_per_celltype[celltype] = embeddings

    


    torch.save(embeddings_per_celltype["NK cells"], os.path.join(args.save_path, "embeddings_NK.pt"))
    torch.save(embeddings_per_celltype["T cells"], os.path.join(args.save_path, "embeddings_T.pt"))
    torch.save(embeddings_per_celltype["Myeloid cells"], os.path.join(args.save_path, "embeddings_Myeloid.pt"))
    torch.save(embeddings_per_celltype["B cells"], os.path.join(args.save_path, "embeddings_B.pt"))
        

if __name__ == "__main__":
    main()