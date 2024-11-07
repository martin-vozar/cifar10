import torch
from torch import nn
from torch.nn import functional as F

from config import ModelConfig
from models.RN import RN

from dataclasses import dataclass
from itertools import product

def get_configsRes04():
    ARCHITECTURES = [
        [RN, 'RN'],
    ]
    NUM_BLOCKS = [
        4,
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
        1e-3,
        3e-4,
        1e-3,
        3e-4,
        3e-5,
        1e-5,
        3e-5,
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
            num_blocks=nb,
            optimizer=opt[0],
            weight_decay=wd,
            activation=act[0],
            lr=lr,
            name=f'{arch[1]}-{nb:02d}_{opt[1]}_lr{lr:.0e}_wd{wd:.0e}_{act[1]}',
        )
        for arch, nb, (opt, lr), wd, act in product(ARCHITECTURES, NUM_BLOCKS, zip(OPTIMIZERS, LRS), WEIGHT_DECAYS, ACTIVATIONS)
    ]
    return configutions