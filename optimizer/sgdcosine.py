#! /usr/bin/python
# -*- encoding: utf-8 -*-

# SGDCosineAnnealing.py
import torch

def Optimizer(parameters, lr=0.1, weight_decay=5e-4, **kwargs):
    print("Initialized SGD with momentum and cosine annealing")
    optimizer = torch.optim.SGD(parameters, lr=lr, momentum=0.9, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    return optimizer, scheduler
