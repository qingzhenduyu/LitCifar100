from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer
import torch
from models.lit_densenet import CIFAR100Model
from datasets.dataset import CIFAR100Datamodule

from pytorch_lightning.plugins.training_type.ddp import DDPPlugin
from pytorch_lightning.utilities.cli import LightningCLI


cli = LightningCLI(
    CIFAR100Model,
    CIFAR100Datamodule,
    save_config_overwrite=True,
    trainer_defaults={"plugins": DDPPlugin(find_unused_parameters=False)},
)
