import math
from typing import List

import pytorch_lightning as pl
import timm
import torch
import torch.nn.functional as F
from timm.loss import LabelSmoothingCrossEntropy
from torch.nn.modules import CrossEntropyLoss
from torchmetrics import Accuracy
from utils.utils import (CombinedScheduler, LinearWarmupScheduler,
                         get_optimizer, initialize_weights, mixup_data, cutmix_data)

import numpy as np

class CIFAR100Model(pl.LightningModule):
    def __init__(
        self,
        model_name: str = 'densenet121',
        pretrained: bool = False,
        learning_rate: float = 0.1,
        momentum: float = 0.9,
        weight_decay: float = 5e-4,
        optimizer_name: str = 'sgd',
        scheduler_name: str = 'step',
        smoothing: float = 0.,
        step_size: int = 60,
        xavier_init: bool = False,
        warmup_steps: int = 5,
        use_mixup: bool = False,
    ):
        super(CIFAR100Model, self).__init__()
        self.save_hyperparameters()
        self.model = timm.create_model(
            model_name, pretrained=pretrained, num_classes=100)

        self.train_accuracy = Accuracy()
        self.val_accuracy = Accuracy()
        self.criterion = CrossEntropyLoss()
        if smoothing != 0:
            self.criterion = LabelSmoothingCrossEntropy(smoothing=smoothing)
        if xavier_init:
            self.model.apply(initialize_weights)
        self.use_mixup = use_mixup

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        
        if self.use_mixup:
            x, y_a, y_b, lam = mixup_data(x, y, alpha=0.2)
            y_hat = self(x)
            loss = lam * self.criterion(y_hat, y_a) + (1 - lam) * self.criterion(y_hat, y_b)
            # if np.random.rand() > 0.5:
            # else:
            #     x, y_a, y_b, lam = cutmix_data(x, y)
            #     y_hat = self(x)
            #     loss = lam * self.criterion(y_hat, y_a) + (1 - lam) * self.criterion(y_hat, y_b)
        else:
            y_hat = self(x)
            loss = self.criterion(y_hat, y)
        self.train_accuracy(y_hat, y)
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', self.train_accuracy, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.val_accuracy(y_hat, y)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', self.val_accuracy, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(), lr=self.hparams.learning_rate, momentum=self.hparams.momentum,
            weight_decay=self.hparams.weight_decay)

        if self.hparams.optimizer_name == 'adam':
            optimizer = torch.optim.Adam(
                self.parameters(), lr=self.hparams.learning_rate
            )
        scheduler = None
        init_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=self.hparams.step_size, gamma=0.1)

        if self.hparams.scheduler_name == 'plateau':
            reduce_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='max', factor=0.1, patience=5, verbose=True)
            scheduler = {
                "scheduler": reduce_scheduler,
                "monitor": "val_acc",
                "interval": "epoch",
                "frequency": self.trainer.check_val_every_n_epoch,
                "strict": True,
            }
        if self.hparams.scheduler_name == 'warmup_step':
            warmup_scheduler = LinearWarmupScheduler(
                optimizer, warmup_steps=5, final_lr=self.hparams.learning_rate)
            scheduler = CombinedScheduler(warmup_scheduler, init_scheduler)

        if self.hparams.scheduler_name == 'warmup_cosine':
            # Define a lambda function for the learning rate schedule
            def lr_lambda(current_step):
                warmup_steps = self.hparams.warmup_steps
                max_epochs = self.trainer.max_epochs

                if current_step < warmup_steps:
                    return float(current_step) / float(max(1, warmup_steps))
                progress = float(current_step - warmup_steps) / \
                    float(max(1, max_epochs - warmup_steps))
                return 0.5 * (1.0 + math.cos(math.pi * progress))

            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        if scheduler is None:
            scheduler = init_scheduler

        return {"optimizer": optimizer, "lr_scheduler": scheduler}
