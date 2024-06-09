import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from typing import List, Optional, Tuple
from timm.data import create_transform


class CIFAR100Datamodule(pl.LightningDataModule):
    def __init__(
        self,
        data_adr: str = "../data/",
        train_batch_size: int = 64,
        eval_batch_size: int = 64,
        num_workers: int = 4,
        img_size: int = 32,
    ) -> None:
        super().__init__()
        self.data_adr = data_adr
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.num_workers = num_workers

        # self.train_transform = transforms.Compose([
        #     transforms.RandomHorizontalFlip(),
        #     transforms.RandomCrop(32, 4),
        #     transforms.ToTensor(),
        #     transforms.Normalize((0.5071, 0.4867, 0.4408),
        #                          (0.2675, 0.2565, 0.2761))
        # ])
        # self.test_transform = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Normalize((0.5071, 0.4867, 0.4408),
        #                          (0.2675, 0.2565, 0.2761))
        # ])
        self.train_transform = create_transform(
            input_size=img_size,
            is_training=True,
            auto_augment='rand-m9-mstd0.5-inc1',
            interpolation='bilinear',
            mean=(0.5071, 0.4867, 0.4408),
            std=(0.2675, 0.2565, 0.2761),
        )

        self.test_transform = create_transform(
            input_size=img_size,
            is_training=False,
            interpolation='bilinear',
            mean=(0.5071, 0.4867, 0.4408),
            std=(0.2675, 0.2565, 0.2761),
        )

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == "fit" or stage is None:
            self.train_dataset = datasets.CIFAR100(
                root=self.data_adr, train=True, download=True, transform=self.train_transform)
            self.val_dataset = datasets.CIFAR100(
                root=self.data_adr, train=False, download=True, transform=self.test_transform)
        if stage == "test" or stage is None:
            self.test_dataset = datasets.CIFAR100(
                root=self.data_adr, train=False, download=True, transform=self.test_transform)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            num_workers=self.num_workers,
            batch_size=self.train_batch_size
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            shuffle=False,
            num_workers=self.num_workers,
            batch_size=self.eval_batch_size
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            shuffle=False,
            num_workers=self.num_workers,
            batch_size=self.eval_batch_size
        )
