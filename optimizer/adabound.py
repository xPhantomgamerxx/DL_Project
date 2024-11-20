#! /usr/bin/python
# -*- encoding: utf-8 -*-

# AdaBound.py
from torch_optimizer import AdaBound  # Requires torch-optimizer

def Optimizer(parameters, lr=0.001, final_lr=0.1, weight_decay=0.0, **kwargs):

    print("Initialized AdaBound optimizer")
    
    return AdaBound(parameters, lr=lr, final_lr=final_lr, weight_decay=weight_decay)