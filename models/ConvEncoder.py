import torch
from torch import nn
from torch.nn import functional as F

class Down(nn.Module):
    def __init__(
        self, 
        activation=nn.ReLU(), 
        in_channels=3,
        out_channels=1,
    ):
        super(Down, self).__init__()    
        
        self.a = activation
        
        self.conv = nn.Conv2d(
            kernel_size=3,
            stride=1,
            padding=1,
            in_channels=in_channels, 
            out_channels=out_channels,
            bias=False,
        )
        self.pool = nn.MaxPool2d(
            kernel_size=3,
            stride=3,
        )
    
        
    def forward(self, x):
        
        x = self.a(self.conv(x))
        x = self.pool(x)
        
        return x
        

class ConvEncoder(nn.Module):
    def __init__(self, in_channels=3, activation=nn.ReLU(), num_classes=10):
        super(ConvEncoder, self).__init__()
        
        channels = [in_channels, 64, 64, 32]
        
        self.down1 = Down(in_channels=channels[0], out_channels=channels[1], activation=activation)
        self.down2 = Down(in_channels=channels[1], out_channels=channels[2], activation=activation)
        self.down3 = Down(in_channels=channels[2], out_channels=channels[3], activation=activation)

        self.output = nn.Linear(32, num_classes)

    def forward(self, x):

        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
                
        x = x.view(x.size(0), -1)
        x = self.output(x)
            
        return x