import torch
from torch import nn
from torch.nn import functional as F

from config import ModelConfig
from models.RN import RN
from models.RR import RR
from models.ResNetC import RN20c
from models.ResNetCI import RN20ci

from dataclasses import dataclass
from itertools import product

def get_configsRRf1():
    ARCHITECTURES = [
        [RR, 'RR'],
        [RN20c, 'RC'],
        [RN20ci, 'RCi']
    ]
    NUM_BLOCKS = [
        20,
    ]
    OPTIMIZERS = [
        [torch.optim.AdamW, 'AdamW'],
        [torch.optim.SGD, 'SGDwM'],
        [torch.optim.SGD, 'SGDwM'],
    ]
    LRS = [
        1e-4,
        1e-1,
        1e-1,
    ]
    WEIGHT_DECAYS = [
        1e-4,
    ]
    ACTIVATIONS = [
        [nn.ReLU(), 'ReLU'],
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
            name=f'{arch[1]}-{nb:02d}_{opt[1]}_lr{lr:.1e}_wd{wd:.1e}_{act[1]}' if arch[1]=='RR' else f'{arch[1]}-20_{opt[1]}_lr{lr:.1e}_wd{wd:.1e}_{act[1]}',
        )
        for (arch, opt, lr), nb,  wd, act in product(zip(ARCHITECTURES, OPTIMIZERS, LRS), NUM_BLOCKS, WEIGHT_DECAYS, ACTIVATIONS)
    ]
    return configutions