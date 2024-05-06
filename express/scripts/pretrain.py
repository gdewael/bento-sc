import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:2048"

from express.data import ExpressDataModule
from express.models import ExpressTransformer
from express.utils.config import Config
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.plugins.environments import LightningEnvironment
from lightning.pytorch import Trainer
import sys

data_file = str(sys.argv[1])
config_path = str(sys.argv[2])
logs_path = str(sys.argv[3])
lr_search_mode = str(sys.argv[4])
ckpt_path = str(sys.argv[5])

config = Config(config_path)
config["data_path"] = data_file

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
    "/".join(logs_path.split("/")[:-1]),
    name=logs_path.split("/")[-1],
)

if lr_search_mode == "True":
    max_steps = 2_501
    val_check_interval = 250
else:
    max_steps = 200_000
    val_check_interval = 5_000

trainer = Trainer(
    accelerator="gpu",
    devices=config.devices,
    strategy="auto",
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

trainer.fit(model, dm, ckpt_path=(None if ckpt_path=="None" else ckpt_path))
