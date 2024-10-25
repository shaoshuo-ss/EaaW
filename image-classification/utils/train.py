# -*- coding: UTF-8 -*-
import torch
import numpy as np


def get_optim(model, optim, lr=0.1, momentum=0, wd=0.0):
    if optim == 'sgd':
        return torch.optim.SGD(model, lr=lr, momentum=momentum, weight_decay=wd)
    elif optim == 'adam':
        return torch.optim.Adam(model, lr=lr, weight_decay=wd)
    elif optim == 'adamw':
        return torch.optim.AdamW(model, lr=lr, weight_decay=wd)
    else:
        exit("Unknown Optimizer!")

