import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:2048"

from bento_sc.data import BentoDataModule
from bento_sc.models import BentoTransformer
from bento_sc.utils.config import Config
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.plugins.environments import LightningEnvironment
from lightning.pytorch import Trainer
import sys

data_file = str(sys.argv[1])
config_path = str(sys.argv[2])
logs_path = str(sys.argv[3])

config = Config(config_path)
config["data_path"] = data_file

dm = BentoDataModule(
    config
)
dm.setup(None)

model = BentoTransformer(
    config
)

callbacks = [
    ModelCheckpoint(every_n_train_steps=5000),
    EarlyStopping(monitor="val_loss", patience=10, mode="min", stopping_threshold=1.225),
]
logger = TensorBoardLogger(
    "/".join(logs_path.split("/")[:-1]),
    name=logs_path.split("/")[-1],
)


max_steps = 100_001
val_check_interval = 500

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

trainer.fit(model, dm)
