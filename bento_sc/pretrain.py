import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:2048"

from bento_sc.data import BentoDataModule
from bento_sc.models import BentoTransformer
from bento_sc.utils.config import Config
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.plugins.environments import LightningEnvironment
from lightning.pytorch import Trainer
import argparse


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
        description="Pre-training script.",
        formatter_class=CustomFormatter,
    )

    parser.add_argument(
        "config_path", type=str, metavar="config_path", help="config_path"
    )
    parser.add_argument("logs_path", type=str, metavar="logs_path", help="logs_path")
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Data file. Overrides value in config file if specified",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate. Overrides value in config file if specified",
    )
    parser.add_argument(
        "--ckpt_path", type=str, default=None, help="Continue from checkpoint"
    )
    parser.add_argument(
        "--tune_mode",
        type=boolean,
        default=False,
        help="Don't pre-train whole model but run small experiment.",
    )

    args = parser.parse_args()

    config = Config(args.config_path)

    if args.data_path is not None:
        config["data_path"] = args.data_path
    if args.lr is not None:
        config["lr"] = args.lr

    dm = BentoDataModule(config)
    dm.setup(None)

    model = BentoTransformer(config)

    callbacks = [
        ModelCheckpoint(every_n_train_steps=5000),
    ]
    logger = TensorBoardLogger(
        "/".join(args.logs_path.split("/")[:-1]),
        name=args.logs_path.split("/")[-1],
    )

    if args.tune_mode:
        max_steps = 2_501
        val_check_interval = 250
    else:
        max_steps = 200_000
        val_check_interval = 5_000
    if ("no_genewise_loss" in config) and (config["no_genewise_loss"] == True):
        strategy = "ddp_find_unused_parameters_true"
    else:
        strategy = "auto"
    trainer = Trainer(
        accelerator="gpu",
        devices=config.devices,
        strategy=strategy,
        plugins=[LightningEnvironment()],
        gradient_clip_val=1,
        max_steps=max_steps,
        val_check_interval=val_check_interval,
        check_val_every_n_epoch=None,
        callbacks=callbacks,
        logger=logger,
        precision="bf16-true",
        use_distributed_sampler=(True if config.return_zeros else False),
    )

    trainer.fit(model, dm, ckpt_path=args.ckpt_path)


if __name__ == "__main__":
    main()
