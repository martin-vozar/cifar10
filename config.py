from torch import nn
from typing import Any, Optional
from dataclasses import dataclass

@dataclass
class ModelConfig:
    architecture: list | nn.Module
    num_blocks: Optional[int]
    optimizer: list | Any
    weight_decay: float
    activation: list | Any
    lr: float
    name: str