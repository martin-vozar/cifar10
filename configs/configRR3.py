import torch
from torch import nn
from torch.nn import functional as F

from config import ModelConfig
from models.RN import RN
from models.RR import RR

from dataclasses import dataclass
from itertools import product

def get_configsRR3():
    ARCHITECTURES = [
        [RR, 'RR'],
    ]
    NUM_BLOCKS = [
        16,
        32,
    ]
    OPTIMIZERS = [
        [torch.optim.AdamW, 'AdamW'],
    ]
    LRS = [
        4e-4,
        3.5e-4,
        3e-4,
        2e-4,
    ]
    WEIGHT_DECAYS = [
        4e-2,
        3.5e-2,
        3e-2,
        2e-2,
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
            name=f'{arch[1]}-{nb:02d}_{opt[1]}_lr{lr:.1e}_wd{wd:.1e}_{act[1]}',
        )
        for arch, nb, opt, (lr, wd), act in product(ARCHITECTURES, NUM_BLOCKS, OPTIMIZERS, zip(LRS, WEIGHT_DECAYS), ACTIVATIONS)
    ]
    return configutions