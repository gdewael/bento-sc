

import torch
import numpy as np
from bento_sc.data import BentoDataModule
from bento_sc.models import BentoTransformer
from lightning.pytorch.plugins.environments import LightningEnvironment
from lightning.pytorch import Trainer
from bento_sc.utils.config import Config
from tqdm import tqdm
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

    else:
        model = BentoTransformer(
            config
        )

        trainer = Trainer(
            accelerator="gpu",
            devices=config.devices,
            strategy="auto",
            plugins=[LightningEnvironment()],
            logger=False,
            enable_checkpointing=False,
            precision="bf16-true",
            use_distributed_sampler=(True if config.return_zeros else False),
        )

        preds = trainer.predict(model, datamodule=dm, ckpt_path=(None if args.approach == "None" else args.approach))
        obs = torch.cat([p[0] for p in preds]).numpy()
        embeds = torch.cat([p[1] for p in preds]).float().numpy()


    np.savez(args.save_path, obs=obs, embeds=embeds)


if __name__ == "__main__":
    main()