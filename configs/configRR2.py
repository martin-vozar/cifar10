import torch
from torch import nn
from torch.nn import functional as F

from config import ModelConfig
from models.RN import RN
from models.RR import RR

from dataclasses import dataclass
from itertools import product

def get_configsRR2():
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
        7e-4,
        5e-4,
        3e-4,
        1e-4,
    ]
    WEIGHT_DECAYS = [
        7e-2,
        5e-2,
        3e-2,
        1e-2,
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