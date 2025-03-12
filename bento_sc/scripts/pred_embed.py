import os
os.environ["OMP_NUM_THREADS"] = "4" # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = "4" # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = "4" # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = "4" # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = "4" # export NUMEXPR_NUM_THREADS=1

import torch
import numpy as np
from bento_sc.data import BentoDataModule
from bento_sc.models import BentoTransformer
from lightning.pytorch.plugins.environments import LightningEnvironment
from lightning.pytorch import Trainer
from bento_sc.utils.config import Config
from tqdm import tqdm
from sklearn.decomposition import IncrementalPCA
import argparse


def main():
    class CustomFormatter(
        argparse.ArgumentDefaultsHelpFormatter, argparse.MetavarTypeHelpFormatter
    ):
        pass

    parser = argparse.ArgumentParser(
        description="Training script for modality prediction.",
        formatter_class=CustomFormatter,
    )

    parser.add_argument("config_path", type=str, metavar="config_path", help="config_path")
    parser.add_argument("approach", type=str, metavar="approach", help="approach")
    parser.add_argument("save_path", type=str, metavar="save_path", help="save_path")
    parser.add_argument("--data_path", type=str, default=None, help="Data file. Overrides value in config file if specified")
    args = parser.parse_args()

    config = Config(args.config_path)

    if args.data_path is not None:
        config["data_path"] = args.data_path

    dm = BentoDataModule(
        config
    )
    dm.setup(None)
    

    if args.approach == "baseline":
        embeds = []
        obs = []
        for batch in tqdm(dm.predict_dataloader()):
            embeds.append(batch["gene_counts"])
            obs.append(batch["0/obs"])

        embeds = torch.cat(embeds).numpy()
        obs = torch.cat(obs).numpy()

    if args.approach == "pca":
        
        ipca = IncrementalPCA(n_components=50)

        for batch in tqdm(dm.predict_dataloader()):
            try:
                ipca.partial_fit(batch["gene_counts"])
            except:
                continue

        embeds = []
        obs = []
        for batch in tqdm(dm.predict_dataloader()):
            embeds.append(torch.tensor(
                ipca.transform(batch["gene_counts"])
            ))
            obs.append(batch["0/obs"])
            

        embeds = torch.cat(embeds).numpy()
        obs = torch.cat(obs).numpy()

    else:
        model = BentoTransformer.load_from_checkpoint(args.approach)

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


if __name__ == "__main__":
    main()