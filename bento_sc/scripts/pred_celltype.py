from bento_sc.data import BentoDataModule
from bento_sc.utils.config import Config
from bento_sc.models import CLSTaskTransformer, BentoTransformer
from bento_sc.baselines import CLSTaskBaseline
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch import Trainer
import sys

config_path = str(sys.argv[1])
approach = str(sys.argv[2])
logs_path = str(sys.argv[3])
no = str(sys.argv[4])
transfer_ct_clf_loss = str(sys.argv[5])



config = Config(config_path)

dm = BentoDataModule(config)

dm.setup(None)

if approach == "baseline":
    model = CLSTaskBaseline(config)

elif approach == "None":
    model = CLSTaskTransformer(config)
else:
    model = CLSTaskTransformer(config)
    pretrained_model = BentoTransformer.load_from_checkpoint(approach)

    pretrained_dict = pretrained_model.state_dict()
    model_dict = model.state_dict()
    pretrained_dict_new = {
        k: v for k, v in pretrained_dict.items() if not k.startswith(("nce_loss", "ct_clf_loss", "loss"))
    }
    if ("ct_clf_loss" in pretrained_dict) and (transfer_ct_clf_loss == "True"):
        pretrained_dict_new["loss"] = pretrained_dict["ct_clf_loss"]

    model_dict.update(pretrained_dict_new)
    model.load_state_dict(model_dict)


val_ckpt = ModelCheckpoint(monitor="val_macroacc", mode="max")
callbacks = [val_ckpt, EarlyStopping(monitor="val_macroacc", patience=20, mode="max")]

logger = TensorBoardLogger(
    logs_path,
    name=no,
)

trainer = Trainer(
    accelerator="gpu",
    devices=config.devices,
    strategy="auto",
    max_steps=500_000,
    val_check_interval=10_000,
    gradient_clip_val=1,
    callbacks=callbacks,
    logger=logger,
    precision="bf16-true",
)

trainer.fit(model, dm.train_dataloader(), dm.val_dataloader())