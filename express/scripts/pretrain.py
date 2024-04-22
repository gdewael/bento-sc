from express.data import ExpressDataModule
from express.models import ExpressTransformer
from express.utils.config import Config
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.plugins.environments import LightningEnvironment
from lightning.pytorch import Trainer
import sys
import os

config_path = str(sys.argv[1])
logs_path = str(sys.argv[2])
no = str(sys.argv[3])
lr_search_mode = str(sys.argv[4])

config = Config(config_path)

dm = ExpressDataModule(
    config
)
dm.setup(None)

model = ExpressTransformer(
    config
)

callbacks = [
    ModelCheckpoint(every_n_train_steps=5000),
]
logger = TensorBoardLogger(
    logs_path,
    name=no,
)

if lr_search_mode == "True":
    max_steps = 1_000
else:
    max_steps = 400_000

trainer = Trainer(
    accelerator="gpu",
    devices=config.devices,
    strategy="auto",
    plugins=[LightningEnvironment()],
    gradient_clip_val=1,
    max_steps=max_steps,
    val_check_interval=5000,
    check_val_every_n_epoch=None,
    callbacks=callbacks,
    logger=logger,
    precision="bf16-true",
    use_distributed_sampler=(True if config.return_zeros else False),
)

trainer.fit(model, dm)