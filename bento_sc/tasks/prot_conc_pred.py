from bento_sc.data import BentoDataModule
from bento_sc.utils.config import Config
from bento_sc.models import CLSTaskTransformer, BentoTransformer
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
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
        description="Fine-tuning script for modality prediction evaluation.",
        formatter_class=CustomFormatter,
    )

    parser.add_argument(
        "config_path", type=str, metavar="config_path", help="config_path"
    )
    parser.add_argument("checkpoint", type=str, metavar="checkpoint", help="checkpoint")
    parser.add_argument("logs_path", type=str, metavar="logs_path", help="logs_path")
    parser.add_argument(
        "--n_workers",
        type=int,
        default=None,
        help="Num workers. Overrides value in config file if specified",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate. Overrides value in config file if specified",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Batch size. Overrides value in config file if specified",
    )
    parser.add_argument(
        "--tune_mode",
        type=boolean,
        default=False,
        help="Don't pre-train whole model but run small experiment.",
    )

    args = parser.parse_args()

    config = Config(args.config_path)

    if args.lr is not None:
        config["lr"] = args.lr
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    if args.n_workers is not None:
        config["n_workers"] = args.n_workers

    dm = BentoDataModule(config)

    dm.setup(None)

    model = CLSTaskTransformer(config)
    pretrained_model = BentoTransformer.load_from_checkpoint(args.checkpoint)

    pretrained_dict = pretrained_model.state_dict()
    model_dict = model.state_dict()
    pretrained_dict_new = {
        k: v
        for k, v in pretrained_dict.items()
        if not k.startswith(("nce_loss", "ct_clf_loss", "loss"))
    }
    model_dict.update(pretrained_dict_new)
    model.load_state_dict(model_dict)

    val_ckpt = ModelCheckpoint(monitor="val_macro_pearson", mode="max")
    callbacks = [
        val_ckpt,
        EarlyStopping(monitor="val_macro_pearson", patience=40, mode="max"),
    ]

    logger = TensorBoardLogger(
        "/".join(args.logs_path.split("/")[:-1]),
        name=args.logs_path.split("/")[-1],
    )

    if args.tune_mode:
        max_steps = int(20_000 // (config.batch_size / 128)) + 1
        val_check_interval = int(400 // (config.batch_size / 128) ** 0.5)
    else:
        max_steps = int(200_000 // (config.batch_size / 128)) + 1
        val_check_interval = int(2_000 // (config.batch_size / 128) ** 0.5)

    trainer = Trainer(
        accelerator="gpu",
        devices=config.devices,
        strategy="auto",
        plugins=[LightningEnvironment()],
        max_steps=max_steps,
        val_check_interval=val_check_interval,
        check_val_every_n_epoch=None,
        gradient_clip_val=1,
        callbacks=callbacks,
        logger=logger,
        precision="bf16-true",
        use_distributed_sampler=(True if config.return_zeros else False),
    )

    trainer.fit(model, dm.train_dataloader(), dm.val_dataloader())

    res = trainer.validate(model, dm.val_dataloader(), ckpt_path="best")
    print(res)
    res = trainer.validate(model, dm.test_dataloader(), ckpt_path="best")
    print(res)


if __name__ == "__main__":
    main()
