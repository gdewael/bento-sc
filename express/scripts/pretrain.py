from express.data import ExpressDataModule
from express.models import ExpressTransformer
from express.utils.config import Config
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch import Trainer
import sys

config_path = str(sys.argv[1])
logs_path = str(sys.argv[2])
no = str(sys.argv[3])

config = Config(config_path)

dm = ExpressDataModule(
    config
)
dm.setup(None)

model = ExpressTransformer(
    config
)


callbacks = [
    ModelCheckpoint(every_n_train_steps=1_00),
]
logger = TensorBoardLogger(
    logs_path,
    name=no,
)

trainer = Trainer(
    accelerator="gpu",
    devices=[1],
    strategy="auto",
    gradient_clip_val=1,
    max_steps=1_50,
    val_check_interval=1_00,
    check_val_every_n_epoch=None,
    callbacks=callbacks,
    logger=logger,
    precision="bf16-true",
)

trainer.fit(model, dm.train_dataloader(), dm.val_sub_dataloader())