from express.data import ExpressDataModule
from express.models import ExpressTransformer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch import Trainer
import sys

config_path = str(sys.argv[1])


dm = ExpressDataModule(
    config_path
)
dm.setup(None)

model = ExpressTransformer(
    config_path
)


callbacks = [
    ModelCheckpoint(every_n_train_steps=10_000),
]
logger = TensorBoardLogger(
    "../logs/",
    name="test",
)

trainer = Trainer(
    accelerator="gpu",
    devices=[0],
    strategy="auto",
    gradient_clip_val=1,
    max_steps=50_000,
    val_check_interval=5_000,
    check_val_every_n_epoch=None,
    callbacks=callbacks,
    logger=logger,
    precision="bf16-mixed",
)

trainer.fit(model, dm.train_dataloader(), dm.val_dataloader())