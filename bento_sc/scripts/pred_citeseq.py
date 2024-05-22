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

config = Config(config_path)

dm = BentoDataModule(
    config
)
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
    model_dict.update(pretrained_dict_new)
    model.load_state_dict(model_dict)

val_ckpt = ModelCheckpoint(monitor="val_loss", mode="min")
callbacks = [val_ckpt, EarlyStopping(monitor="val_loss", patience=10, mode="min")]

logger = TensorBoardLogger(
    logs_path,
    name=no,
)

trainer = Trainer(
    accelerator="gpu",
    devices=config.devices,
    strategy="auto",
    max_epochs=100,
    gradient_clip_val=1,
    callbacks=callbacks,
    logger=logger,
    precision="bf16-true",
)

trainer.fit(model, dm.train_dataloader(), dm.val_dataloader())