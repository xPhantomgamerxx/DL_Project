#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import torchvision.models as models

def MainModel(nOut=256, **kwargs):
    model = models.densenet121(weights=None)  # No pretrained weights
    num_features = model.classifier.in_features
    model.classifier = torch.nn.Linear(num_features, nOut)
    return model
