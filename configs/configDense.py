import torch
from torch import nn
from torch.nn import functional as F

from config import ModelConfig
from models.naive import Naive

from dataclasses import dataclass
from itertools import product

def get_configsDense():
    ARCHITECTURES = [
        [Naive, 'Dense'],
    ]
    OPTIMIZERS = [
        [torch.optim.SGD, 'SGD'],
        [torch.optim.SGD, 'SGD'],
        [torch.optim.SGD, 'SGDwM'],
        [torch.optim.SGD, 'SGDwM'],
        [torch.optim.Adam, 'Adam'],
        [torch.optim.Adam, 'Adam'],
        [torch.optim.AdamW, 'AdamW'],
        [torch.optim.AdamW, 'AdamW'],
    ]
    LRS = [
        5e-3,
        1e-3,
        1e-3,
        5e-4,
        5e-5,
        1e-5,
        5e-5,
        1e-5,
    ]
    WEIGHT_DECAYS = [
        1e-4,
    ]
    ACTIVATIONS = [
        [nn.LeakyReLU(), 'LeakyReLU'],
        # [nn.Tanh(), 'Tanh'],
    ]

    configutions = [
        ModelConfig(
            architecture=arch[0],
            optimizer=opt[0],
            weight_decay=wd,
            activation=act[0],
            lr=lr,
            name=f'{arch[1]}_{opt[1]}_lr{lr:.0e}_wd{wd:.0e}_{act[1]}',
        )
        for arch, (opt, lr), wd, act in product(ARCHITECTURES, zip(OPTIMIZERS, LRS), WEIGHT_DECAYS, ACTIVATIONS)
    ]
    return configutions