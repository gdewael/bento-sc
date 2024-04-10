from express.data import ExpressDataModule
from express.models import PerturbTransformer
from express.baselines import PerturbMixer
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch import Trainer
import sys

config_path = str(sys.argv[1])
baseline = str(sys.argv[2])
logs_path = str(sys.argv[3])
no = str(sys.argv[4])

dm = ExpressDataModule(
    config_path
)
dm.setup(None)

if baseline == "baseline":
    model = PerturbMixer(config_path)
else:
    model = PerturbTransformer(
        config_path
    )

val_ckpt = ModelCheckpoint(monitor="val_loss", mode="min")
callbacks = [val_ckpt, EarlyStopping(monitor="val_loss", patience=10, mode="min")]

logger = TensorBoardLogger(
    logs_path,
    name=no,
)

trainer = Trainer(
    accelerator="gpu",
    devices=[0],
    strategy="auto",
    max_epochs=10,
    gradient_clip_val=1,
    callbacks=callbacks,
    logger=logger,
    precision="bf16-mixed",
)

trainer.fit(model, dm.train_dataloader(), dm.val_dataloader())