import torch
from torch import nn
from torch.nn import functional as F

from models.ConvEncoder import ConvEncoder
from models.RRBlock import RRBlock

class RR(nn.Module):
    def __init__(
        self, 
        activation=nn.ReLU(), 
        in_channels=3,
        out_channels=1,
        num_classes=10,
        num_blocks=4,
    ):
        
        super(RR, self).__init__()    
        
        self.width = 16
        
        self.drop = nn.Dropout2d(p=1/3)
        self.input = nn.Conv2d(
            in_channels=in_channels,
            out_channels=self.width,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )

        self.blocks = nn.ModuleList(
            [RRBlock(
                in_channels=self.width, 
                out_channels=self.width, 
                activation=activation) 
             for _ in range(num_blocks)
            ]
        )
        
        self.encoder = ConvEncoder(in_channels=self.width, num_classes=num_classes, activation=activation)

    def forward(self, x):
        
        x = self.drop(x)  
        z = x.clone()
        
        x = self.input(x)
        
        for block in self.blocks:
            x = block(torch.cat([x, z], dim=1))
            
        x = self.encoder(x)
        
        return x