#! /usr/bin/python
# -*- encoding: utf-8 -*-

# LARS.py
from torch_optimizer import LARS

def Optimizer(parameters, lr=0.1, weight_decay=0.0, **kwargs):
    print("Initialized LARS optimizer")
    return LARS(parameters, lr=lr, weight_decay=weight_decay)
