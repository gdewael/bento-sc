from bento_sc.data import BentoDataModule
from bento_sc.utils.config import Config
from bento_sc.models import CLSTaskTransformer, BentoTransformer
from bento_sc.baselines import CLSTaskBaseline
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
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
        description="Training script for celltype clf.",
        formatter_class=CustomFormatter,
    )

    parser.add_argument("config_path", type=str, metavar="config_path", help="config_path")
    parser.add_argument("approach", type=str, metavar="approach", help="approach")
    parser.add_argument("logs_path", type=str, metavar="logs_path", help="logs_path")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate. Overrides value in config file if specified")
    parser.add_argument("--transfer_ct_clf_loss", type=boolean, default=False, help="Whether to transfer ct clf loss from pre-trained model.")

    if args.lr is not None:
        config["lr"] = args.lr

    args = parser.parse_args()

    config = Config(args.config_path)

    dm = BentoDataModule(config)

    dm.setup(None)

    if args.approach == "baseline":
        model = CLSTaskBaseline(config)

    elif args.approach == "None":
        model = CLSTaskTransformer(config)
    else:
        model = CLSTaskTransformer(config)
        pretrained_model = BentoTransformer.load_from_checkpoint(args.approach)

        pretrained_dict = pretrained_model.state_dict()
        model_dict = model.state_dict()
        pretrained_dict_new = {
            k: v for k, v in pretrained_dict.items() if not k.startswith(("nce_loss", "ct_clf_loss", "loss"))
        }
        if ("ct_clf_loss" in pretrained_dict) and (args.transfer_ct_clf_loss == True):
            pretrained_dict_new["loss"] = pretrained_dict["ct_clf_loss"]

        model_dict.update(pretrained_dict_new)
        model.load_state_dict(model_dict)


    val_ckpt = ModelCheckpoint(monitor="val_macroacc", mode="max")
    callbacks = [val_ckpt, EarlyStopping(monitor="val_macroacc", patience=20, mode="max")]

    logger = TensorBoardLogger(
        "/".join(args.logs_path.split("/")[:-1]),
        name=args.logs_path.split("/")[-1],
    )

    trainer = Trainer(
        accelerator="gpu",
        devices=config.devices,
        strategy="auto",
        max_steps=500_000,
        val_check_interval=2_000,
        gradient_clip_val=1,
        callbacks=callbacks,
        logger=logger,
        precision="bf16-true",
    )

    trainer.fit(model, dm.train_dataloader(), dm.val_dataloader())


if __name__ == "__main__":
    main()