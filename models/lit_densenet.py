import timm
import torch
import pytorch_lightning as pl
import torch.nn.functional as F
from torchmetrics import Accuracy
from typing import List


class CIFAR100Model(pl.LightningModule):
    def __init__(self, model_name: str, learning_rate: float, momentum: float, weight_decay: float, milestones: List[int],):
        super(CIFAR100Model, self).__init__()
        self.save_hyperparameters()
        self.model = timm.create_model(
            model_name, pretrained=False, num_classes=100)
        self.train_accuracy = Accuracy()
        self.val_accuracy = Accuracy()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.train_accuracy(y_hat, y)
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', self.train_accuracy, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.val_accuracy(y_hat, y)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', self.val_accuracy, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(), lr=self.hparams.learning_rate, momentum=self.hparams.momentum,
            weight_decay=self.hparams.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=60, gamma=0.1)

        return {"optimizer": optimizer, "lr_scheduler": scheduler}
