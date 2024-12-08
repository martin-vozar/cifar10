import torch
from torch import nn
from torch.nn import functional as F

from config import ModelConfig
from models.RN import RN
from models.RR import RR

from dataclasses import dataclass
from itertools import product

def get_configsRR0():
    ARCHITECTURES = [
        [RN, 'RN'],
        [RR, 'RR']
    ]
    NUM_BLOCKS = [
        16,
    ]
    OPTIMIZERS = [
        [torch.optim.AdamW, 'AdamW'],
    ]
    LRS = [
        1e-4,
        1e-4,
        5e-5,
        5e-5,
    ]
    WEIGHT_DECAYS = [
        1e-2,
        5e-3,
        5e-3,
        1e-3,
    ]
    ACTIVATIONS = [
        [nn.LeakyReLU(), 'LeakyReLU'],
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
        for arch, nb, opt, (lr, wd), act in product(ARCHITECTURES, NUM_BLOCKS, OPTIMIZERS, zip(LRS, WEIGHT_DECAYS), ACTIVATIONS)
    ]
    return configutions