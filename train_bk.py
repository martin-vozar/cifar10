import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.nn import functional as F

import tqdm
from typing import Any
from dataclasses import dataclass
from itertools import product

from models.naive import Naive
from models.naive import NaiveMin
from data import get_split_dls

if torch.cuda.is_available():
    DEVICE = "cuda"
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
else:
    DEVICE = "cpu"
    torch.use_deterministic_algorithms(True)

@dataclass
class Config:
    architecture: list |nn.Module
    optimizer: list | Any
    weight_decay: float
    activation: list | Any
    lr: float
    name: str

MAX_EPOCHS = 200
NUM_WORKERS = 4
DOWNLOAD = False # just avoiding unnecessary outputs sying everything is cool

ARCHITECTURES = [
    [Naive, 'Dense'],
    [NaiveMin, 'DnMin'],
]
OPTIMIZERS = [
    [torch.optim.SGD, 'SGD'],
    [torch.optim.SGD, 'SGDwM'],
    [torch.optim.Adam, 'Adam'],
    # [torch.optim.Adam, 'Adam'],
    # [torch.optim.AdamW, 'AdamW'],
    # [torch.optim.AdamW, 'AdamW'],
]
LRS = [
    1e-3,
    1e-3,
    1e-3,
    # 1e-4,
    # 1e-3,
    # 1e-4,
]
WEIGHT_DECAYS = [
    1e-4,
]
ACTIVATIONS = [
    [nn.ReLU(), 'ReLU'],
]

configutions = [
    Config(
        architecture=arch[0],
        optimizer=opt[0],
        weight_decay=wd,
        activation=act[0],
        lr=lr,
        name=f'{arch[1]}_{opt[1]}_wd{wd}_{act[1]}_lr{lr:.0e}',
    )
    for arch, (opt, lr), wd, act in product(ARCHITECTURES, zip(OPTIMIZERS, LRS), WEIGHT_DECAYS, ACTIVATIONS)
]

[
    print(cfg.name) 
    for cfg in configutions
]

def main():
    train_loader, test_loader, classes = get_split_dls(
        num_workers=NUM_WORKERS, download=DOWNLOAD
    )
    batches_train = len(train_loader)
    batches_test = len(test_loader)
    NUM_CLASSES = len(classes)
    
    every_n_steps_train = len(train_loader) // 5
    if every_n_steps_train == 0:
        every_n_steps_train = 1
    
    every_n_steps_test = len(test_loader) // 5
    if every_n_steps_test == 0:
        every_n_steps_test = 1
    
    criterion = nn.CrossEntropyLoss(reduction='mean')
    
    # Wrap as some ModelConfig__init__ method
    nets = []
    opts = []
    loss_t = []
    loss_v = []
    tstep_log = []
    tacc_log = []
    tloss_log = []
    vstep_log = []
    vacc_log = []
    vloss_log = []
    for cfg in configutions:
        nets.append(cfg.architecture(cfg.activation, NUM_CLASSES).to(DEVICE))
        if cfg.name.split('_')[1]=="SGDwM":
            opts.append(cfg.optimizer(nets[-1].parameters(), lr=cfg.lr, momentum=0.9, weight_decay=cfg.weight_decay))
        else:
            opts.append(cfg.optimizer(nets[-1].parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay))    
        loss_t.append(0)
        loss_v.append(0)
        tstep_log.append([])
        tloss_log.append([])
        tacc_log.append([])
        vstep_log.append([])
        vloss_log.append([])
        vacc_log.append([])

    for epoch in range(MAX_EPOCHS):
        correct = [0]*len(nets)
        total = [0]*len(nets)
        for step, (x, y) in tqdm.tqdm(enumerate(train_loader), total=batches_train):
            for it, (net, opt) in enumerate(zip(nets, opts)):
                x = x.to(DEVICE, dtype=torch.float32) 
                y = y.to(DEVICE)               
                                
                xi = net(x)
                loss = criterion(xi, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1, norm_type=2)
                opt.step()
                
                loss_t[it] += loss.item()
                
                _, predicted = xi.max(1)
                total[it] += y.size(0)
                correct[it] += predicted.eq(y).sum().item()
                
            if (step+1)%every_n_steps_train==0:
                for it in range(len(nets)):
                    tstep_log[it].append(epoch+step/len(train_loader))
                    tloss_log[it].append(loss_t[it]/every_n_steps_train)
                    loss_t[it] = 0
                    tacc_log[it].append(correct[it]/total[it])
                    correct[it] = 0
                    total[it] = 0
            ### accumulate loss for evaluation -> save as best_{}.pt
        
        correct = [0]*len(nets)
        total = [0]*len(nets)                    
        for step, (x, y) in tqdm.tqdm(enumerate(test_loader), total=batches_test):
            for it, (net, opt) in enumerate(zip(nets, opts)):
                net.eval()
                with torch.no_grad():
                    x = x.to(DEVICE, dtype=torch.float32)    
                    y = y.to(DEVICE)                         
                    xi = net(x)
                    loss = criterion(xi, y)
                    loss_v[it] += loss.item()
                
                    _, predicted = xi.max(1)
                    total[it] += y.size(0)
                    correct[it] += predicted.eq(y).sum().item()
                net.train()
                
            if (step+1)%every_n_steps_test==0:
                for it in range(len(nets)):
                    vstep_log[it].append(epoch+step/len(test_loader))
                    vloss_log[it].append(loss_v[it]/every_n_steps_test)
                    loss_v[it] = 0
                    if total!=0:
                        vacc_log[it].append(correct[it]/total[it])
                    else:
                        vacc_log[it].append(0)
                    correct[it] = 0
                    total[it] = 0
        
        plt.figure(figsize=(12,6))
        for it, cfg in enumerate(configutions):    
            plt.scatter(tstep_log[it], tacc_log[it], s=2, label=f'{cfg.name}')
            plt.plot(tstep_log[it], tacc_log[it], lw=1)
            
        plt.title('Train set: Accuracy')
        plt.grid(which='both')
        plt.xlim(0, 1.6*MAX_EPOCHS)
        plt.ylim(-0.05, 1.05)
        # plt.legend(loc='upper right')
        plt.legend(bbox_to_anchor=(1.0, 0.6))
        plt.savefig('./pngs/train_acc.png')
        plt.close("all")

        plt.figure(figsize=(12,6))        
        for it, cfg in enumerate(configutions):    
            plt.scatter(vstep_log[it], vacc_log[it], s=2, label=f'{cfg.name}')
            plt.plot(vstep_log[it], vacc_log[it], lw=1)

        plt.title('Test set: Accuracy')
        plt.grid(which='both')
        plt.xlim(0, 1.6*MAX_EPOCHS)
        plt.ylim(-0.05, 1.05)
        
        # plt.legend(loc='upper right')
        plt.legend(bbox_to_anchor=(1.0, 0.6))
        plt.savefig('./pngs/test_acc.png')
        plt.close("all")        

        plt.figure(figsize=(12,6))
        for it, cfg in enumerate(configutions):    
            plt.scatter(tstep_log[it], tloss_log[it], s=2, label=f'{cfg.name}')
            plt.plot(tstep_log[it], tloss_log[it], lw=1)
            
        plt.title('Train set: CrossEntropy')
        plt.grid(which='both')
        plt.yscale('log')
        plt.xlim(0, 1.6*MAX_EPOCHS)
        # plt.ylim(top=1)
        
        # plt.legend(loc='upper right')
        plt.legend(bbox_to_anchor=(1.0, 1))
        plt.savefig('./pngs/train_loss.png')
        plt.close("all")
        
        plt.figure(figsize=(12,6))
        for it, cfg in enumerate(configutions):    
            plt.scatter(vstep_log[it], vloss_log[it], s=2, label=f'{cfg.name}')
            plt.plot(vstep_log[it], vloss_log[it], lw=1)

        plt.title('Test set: CrossEntropy')
        plt.grid(which='both')
        plt.yscale('log')
        plt.xlim(0, 1.6*MAX_EPOCHS)
        # plt.ylim(top=1)
        
        # plt.legend(loc='upper right')
        plt.legend(bbox_to_anchor=(1.0, 1))
        plt.savefig('./pngs/test_loss.png')
        plt.close("all")
            
if __name__=="__main__":
    main()