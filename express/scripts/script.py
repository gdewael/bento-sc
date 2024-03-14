import h5torch
import numpy as np
from express.data import ExpressDataModule
from express.models import ExpressTransformer
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch import Trainer
import sys


data = str(sys.argv[1])
bins = str(sys.argv[2])
input_processing = str(sys.argv[3])

if input_processing != "None":
    input_processing = input_processing.split(",")
else:
    input_processing = []


dm = ExpressDataModule(
    data,
    bins,
    batch_size=64,
    input_processing=input_processing,
)
dm.setup(None)

model = ExpressTransformer(
    discrete_input=(True if input_processing == ["bin"] else False),
    n_discrete_tokens=len(np.loadtxt(bins)) - 1,
    dim=256,
    depth=8,
    dropout=0.15,
)

callbacks = [
    ModelCheckpoint(every_n_train_steps=10_000),
]
logger = TensorBoardLogger(
    "./",
    name="_".join(input_processing),
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
