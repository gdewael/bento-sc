from express.data import ExpressDataModule
from express.utils.config import Config
from express.models import CLSTaskTransformer, ExpressTransformer
from express.baselines import CLSTaskBaseline
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch import Trainer
import sys

config_path = str(sys.argv[1])
checkpoint = str(sys.argv[2])
transfer_ct_clf_loss = str(sys.argv[5])
logs_path = str(sys.argv[3])
no = str(sys.argv[4])


if checkpoint == "baseline":
    config = Config(config_path)

    model = CLSTaskBaseline(config)
elif checkpoint == "None":
    config = Config(config_path)
    model = CLSTaskTransformer(
        config
    )
else:
    pretrained_model = ExpressTransformer.load_from_checkpoint(checkpoint)
    config_pretrained = pretrained_model.hparams.config
    print(
        "Pretrained model used. Ignoring following keys in config: " +
        ", ".join([
            "input_processing",
            "return_zeros",
            "discrete_input",
            "n_discrete_tokens",
            "gate_input",
            "pseudoquant_input",
            "dim",
            "depth",
            "dropout",
            "n_genes",

        ]))
    config = Config(config_path)
    config.change_keys(
        input_processing = config_pretrained.input_processing,
        return_zeros = config_pretrained.return_zeros,
        discrete_input = config_pretrained.discrete_input,
        n_discrete_tokens = config_pretrained.n_discrete_tokens,
        gate_input = config_pretrained.gate_input,
        pseudoquant_input = config_pretrained.pseudoquant_input,
        dim = config_pretrained.dim,
        depth = config_pretrained.depth,
        dropout = config_pretrained.dropout,
        n_genes = config_pretrained.n_genes,
        )
    model = CLSTaskTransformer(
        config
    )
    pretrained_dict = pretrained_model.state_dict()
    model_dict = model.state_dict()
    pretrained_dict_new = {
        k: v for k, v in pretrained_dict.items() if not k.startswith(("nce_loss", "ct_clf_loss", "loss"))
    }
    if ("ct_clf_loss" in pretrained_dict) and (transfer_ct_clf_loss == "True"):
        pretrained_dict_new["loss"] = pretrained_dict["ct_clf_loss"]

    model_dict.update(pretrained_dict_new)
    model.load_state_dict(model_dict)


dm = ExpressDataModule(
    config
)
dm.setup(None)

config.print_used_keys(dm.config_used | model.config_used)

val_ckpt = ModelCheckpoint(monitor="val_loss", mode="min")
callbacks = [val_ckpt, EarlyStopping(monitor="val_loss", patience=10, mode="min")]

logger = TensorBoardLogger(
    logs_path,
    name=no,
)

trainer = Trainer(
    accelerator="gpu",
    devices=[2],
    strategy="auto",
    max_steps=500_000,
    val_check_interval=10_000,
    gradient_clip_val=1,
    callbacks=callbacks,
    logger=logger,
    precision="bf16-true",
)

trainer.fit(model, dm.train_dataloader(), dm.val_sub_dataloader())