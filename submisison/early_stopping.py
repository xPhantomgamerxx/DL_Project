#!/usr/bin/python
#-*- coding: utf-8 -*-

import numpy as np
import torch

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0, path='exps/checkpoint.pt'):
        """
        Args:
            patience (int): Number of epochs with no improvement after which training will be stopped.
            min_delta (float): Minimum change in the monitored quantity to qualify as an improvement.
            path (str): Path to save the best model checkpoint.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.path = path
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_loss = np.Inf

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decreases."""
        torch.save(model.state_dict(), self.path)
        self.best_loss = val_loss
