from bento_sc.utils.config import Config
from bento_sc.data import BentoDataModule
from bento_sc.models import BentoTransformer
from bento_sc.utils.metrics import pearson_batch_masked
from lightning.pytorch.plugins.environments import LightningEnvironment
from lightning.pytorch import Trainer
import torch
import numpy as np
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
        description="Training script for reconstruction.",
        formatter_class=CustomFormatter,
    )

    parser.add_argument("config_path", type=str, metavar="config_path", help="config_path")
    parser.add_argument("approach", type=str, metavar="approach", help="approach (model)")
    parser.add_argument("--data_path", type=str, default=None, help="Data file. Overrides value in config file if specified")
    parser.add_argument("--clf_output", type=boolean, default=True, help="If the model produced count predictions through multi-class clf.")
    parser.add_argument("--model_preds_logp1", type=boolean, default=False, help="Set to to true if the pre-trained model predicted log transformed counts.")


    args = parser.parse_args()



    config = Config(args.config_path)
    if args.data_path is not None:
        config["data_path"] = args.data_path

    dm = BentoDataModule(
        config
    )
    dm.setup(None)


    model = BentoTransformer(
        config
    )
    model = model.eval()

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

    counts = [p[-2] for p in preds]
    trues = [p[-1] for p in preds]

    if args.clf_output:
        for temp in [0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100]:
            pearsons = []
            for batch_ix in range(len(counts)):
                    
                multiplier = torch.arange(counts[batch_ix].shape[-1])[None, None, :]

                predicted_as_count = (torch.nn.functional.softmax(
                    counts[batch_ix].float()*temp, -1
                ) * multiplier).sum(-1)

                true_count = trues[batch_ix].float()

                
                pearsons.append(pearson_batch_masked(predicted_as_count, true_count).numpy())

            print(temp, np.concatenate(pearsons).mean())
    else:
        pearsons = []
        for batch_ix in range(len(counts)):
            if args.model_preds_logp1:
                predicted_as_count = torch.expm1(counts[batch_ix].float())
            else:
                predicted_as_count = counts[batch_ix].float()
            true_count = trues[batch_ix].float()
            pearsons.append(pearson_batch_masked(predicted_as_count, true_count).numpy())
        print(np.concatenate(pearsons).mean())


if __name__ == "__main__":
    main()