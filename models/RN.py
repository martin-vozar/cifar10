import torch
from torch import nn
from torch.nn import functional as F

from models.ConvEncoder import ConvEncoder
from models.ResBlock import ResBlock

class RN(nn.Module):
    def __init__(
        self, 
        activation=nn.ReLU(), 
        in_channels=3,
        out_channels=1,
        num_classes=10,
        num_blocks=4,
    ):
        width = 16
        
        super(RN, self).__init__()    
        
        self.input = nn.Conv2d(
            in_channels=in_channels,
            out_channels=width,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        
        self.r1 = ResBlock(in_channels=width, out_channels=width, activation=activation)
        self.r2 = ResBlock(in_channels=width, out_channels=width, activation=activation)
        self.r3 = ResBlock(in_channels=width, out_channels=width, activation=activation)
        self.r4 = ResBlock(in_channels=width, out_channels=width, activation=activation)
        
        self.blocks = nn.ModuleList(
            [ResBlock(
                in_channels=width, 
                out_channels=width, 
                activation=activation) 
             for _ in range(num_blocks)
            ]
        )
        
        
        self.encoder = ConvEncoder(in_channels=width, num_classes=num_classes, activation=activation)

    def forward(self, x):
        
        x = self.input(x)
        
        for block in self.blocks:
            x = block(x)
        
        x = self.encoder(x)
        
        return x