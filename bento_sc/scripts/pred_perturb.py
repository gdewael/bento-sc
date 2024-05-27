from bento_sc.data import BentoDataModule
from bento_sc.models import PerturbTransformer, BentoTransformer
from bento_sc.baselines import PerturbBaseline
from bento_sc.utils.config import Config
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch import Trainer
import argparse

class CustomFormatter(
    argparse.ArgumentDefaultsHelpFormatter, argparse.MetavarTypeHelpFormatter
):
    pass

def main():
    parser = argparse.ArgumentParser(
        description="Training script for post-perturbation expression prediction.",
        formatter_class=CustomFormatter,
    )

    parser.add_argument("config_path", type=str, metavar="config_path", help="config_path")
    parser.add_argument("approach", type=str, metavar="approach", help="approach")
    parser.add_argument("logs_path", type=str, metavar="logs_path", help="logs_path")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate. Overrides value in config file if specified")
    parser.add_argument("--init_factor", type=float, default=None, help="init_factor. Overrides value in config file if specified")

    args = parser.parse_args()

    config = Config(args.config_path)

    if args.lr is not None:
        config.lr = args.lr
    if args.init_factor is not None:
        config.perturb_init_factor = args.init_factor
    

    dm = BentoDataModule(
        config
    )
    dm.setup(None)

    if args.approach == "baseline":
        model = PerturbBaseline(config)
    elif args.approach == "None":
        model = PerturbTransformer(config)
    else:
        model = PerturbTransformer(config)
        pretrained_model = BentoTransformer.load_from_checkpoint(args.approach)

        pretrained_dict = pretrained_model.state_dict()
        model_dict = model.state_dict()
        pretrained_dict_new = {
            k: v for k, v in pretrained_dict.items() if not k.startswith(("nce_loss", "ct_clf_loss", "loss"))
        }
        model_dict.update(pretrained_dict_new)
        model.load_state_dict(model_dict)

    val_ckpt = ModelCheckpoint(monitor="val_loss", mode="min")
    callbacks = [val_ckpt, EarlyStopping(monitor="val_loss", patience=10, mode="min")]

    logger = TensorBoardLogger(
        "/".join(args.logs_path.split("/")[:-1]),
        name=args.logs_path.split("/")[-1],
    )

    trainer = Trainer(
        accelerator="gpu",
        devices=config.devices,
        strategy="auto",
        max_epochs=200,
        gradient_clip_val=1,
        callbacks=callbacks,
        logger=logger,
        precision="bf16-true",
    )

    trainer.fit(model, dm.train_dataloader(), dm.val_dataloader())