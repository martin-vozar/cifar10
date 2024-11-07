import torch
from torch import nn
from torch.nn import functional as F
    
class Naive(nn.Module):
    def __init__(self, activation=nn.ReLU(), num_clasees=10):
        super(Naive, self).__init__()
        
        self.act = activation
        self.drop = nn.Dropout(p=0.3)
        
        self.input = nn.Linear(3072, 256)
        
        self.lin01 = nn.Linear(256, 256)
        self.lin02 = nn.Linear(256, 256)
        self.lin03 = nn.Linear(256, 256)
        self.lin04 = nn.Linear(256, 256)
        self.lin05 = nn.Linear(256, 256)
        self.lin06 = nn.Linear(256, 256)
        self.lin07 = nn.Linear(256, 256)
        self.lin08 = nn.Linear(256, 256)
        
        self.output = nn.Linear(256, num_clasees)
        
    def forward(self, x):
        x = x.view(x.shape[0], -1)
        
        x = self.act(self.input(x))
        
        x = self.act(self.lin01(x))
        x = self.act(self.lin02(x))
        x = self.act(self.lin03(x))
        x = self.act(self.lin04(x))
        x = self.act(self.lin05(x))
        x = self.act(self.lin06(x))
        x = self.act(self.lin07(x))
        x = self.act(self.lin08(x))
        
        x = self.output(x)
        
        return x