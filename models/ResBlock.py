import torch
from torch import nn
from torch.nn import functional as F

class ResBlock(nn.Module):
    def __init__(
        self, 
        activation=nn.ReLU(), 
        in_channels=3,
        out_channels=1,
    ):
        super(ResBlock, self).__init__()    
        
        self.a = activation
        
        self.conv1 = nn.Conv2d(
            kernel_size=3,
            stride=1,
            padding=1,
            in_channels=in_channels, 
            out_channels=2*out_channels,
            bias=False,
        )
        
        self.bn1 = nn.BatchNorm2d(2*out_channels)

        self.conv2 = nn.Conv2d(
            kernel_size=3,
            stride=1,
            padding=1,
            in_channels=2*out_channels, 
            out_channels=out_channels,
            bias=False,
        )
        
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        
        z = self.a(self.bn1(self.conv1(x)))
        z = self.bn2(self.conv2(z))
        
        x = x + z
        
        return x