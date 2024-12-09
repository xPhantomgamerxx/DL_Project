#! /usr/bin/python
# -*- encoding: utf-8 -*-

# EfficientNetB0.py
import torch
import torchvision.models as models

def MainModel(nOut=256, **kwargs):
    model = models.efficientnet_b0(weights=None)  # No pretrained weights
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, nOut)
    return model
