import json
import os
from pathlib import Path
from typing import Iterator

import torch
import torch.nn as nn
from torch.optim import SGD, Adam, AdamW, Optimizer
from torch.optim.lr_scheduler import (CosineAnnealingLR, MultiStepLR, StepLR,
                                      _LRScheduler)
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import numpy as np


def get_optimizer(optimizer_name: str, model_parameters: torch.nn.Parameter, learning_rate: float, weight_decay: float, **kwargs) -> Optimizer:
    """
    return specific optimizer based on optimizer_name like adam, sgd, adamw, etc.

        Parameters
        ----------
        optimizer_name : str
        model_parameters: Iterator[torch.nn.Parameter]
        learning_rate: float
        weight_decay: float

    Return
        -------
        Optimizer
    """
    optimizers = {
        'adam': Adam(params=model_parameters, lr=learning_rate, weight_decay=weight_decay, **kwargs),
        'sgd': SGD(params=model_parameters, lr=learning_rate, weight_decay=weight_decay, **kwargs),
        'adamw': AdamW(params=model_parameters, lr=learning_rate, weight_decay=weight_decay, **kwargs)
    }
    return optimizers[optimizer_name.lower()]


def get_scheduler(scheduler_name: str, optimizer: Optimizer, **kwargs):
    schedulers = {
        'step': StepLR(optimizer, **kwargs),
        'multistep': MultiStepLR(optimizer, **kwargs),
        'cosine': CosineAnnealingLR(optimizer, **kwargs)
    }
    return schedulers[scheduler_name.lower()]

def initialize_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)



class LinearWarmupScheduler(_LRScheduler):
    def __init__(self, optimizer: Optimizer, warmup_steps: int, final_lr: float, last_epoch: int = -1):
        self.warmup_steps = warmup_steps
        self.final_lr = final_lr
        super(LinearWarmupScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            return [self.final_lr * (self.last_epoch + 1) / self.warmup_steps for _ in self.optimizer.param_groups]
        else:
            return [group['lr'] for group in self.optimizer.param_groups]

class CombinedScheduler:
    def __init__(self, warmup_scheduler, main_scheduler):
        self.warmup_scheduler = warmup_scheduler
        self.main_scheduler = main_scheduler

    def step(self, epoch=None, metrics=None):
        if self.warmup_scheduler.last_epoch < self.warmup_scheduler.warmup_steps:
            self.warmup_scheduler.step(epoch)
        else:
            self.main_scheduler.step(epoch if epoch is not None else self.warmup_scheduler.last_epoch - self.warmup_scheduler.warmup_steps, metrics)

    def state_dict(self):
        return {
            "warmup_scheduler": self.warmup_scheduler.state_dict(),
            "main_scheduler": self.main_scheduler.state_dict()
        }

    def load_state_dict(self, state_dict):
        self.warmup_scheduler.load_state_dict(state_dict["warmup_scheduler"])
        self.main_scheduler.load_state_dict(state_dict["main_scheduler"])


def mixup_data(x, y, alpha=1.0):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda()

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def cutmix_data(x, y, alpha=1.0):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda()

    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]

    y_a, y_b = y, y[index]
    return x, y_a, y_b, lam

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int32(W * cut_rat)
    cut_h = np.int32(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2
