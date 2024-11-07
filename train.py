import os
import tqdm
from termcolor import cprint

import torch
from torch import nn

from configs.configRN import get_configsRN
from configs.configRes04 import get_configsRes04
from configs.configDense import get_configsDense
from configs.configConv import get_configsConv
from log2png import log2png
from data import get_split_dls

if torch.cuda.is_available():
    DEVICE = "cuda"
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
else:
    DEVICE = "cpu"
    torch.use_deterministic_algorithms(True)
    

###############################################################################

MOMENTUM = 0.96

MAX_EPOCHS = 100
NUM_WORKERS = 4
DOWNLOAD = True

configurations = get_configsRN() ###########################################

[
    print(cfg.name) 
    for cfg in configurations
]

###############################################################################

def main():
    train_loader, test_loader, classes = get_split_dls(
        num_workers=NUM_WORKERS, download=DOWNLOAD
    )
    batches_train = len(train_loader)
    batches_test = len(test_loader)
    NUM_CLASSES = len(classes)
    
    every_n_steps_train = len(train_loader) // 2
    if every_n_steps_train == 0:
        every_n_steps_train = 1
    
    every_n_steps_test = len(test_loader) // 2 # if ==1 => accumulated loss for test set
    if every_n_steps_test == 0:
        every_n_steps_test = 1
    
    criterion = nn.CrossEntropyLoss(reduction='mean')
    
    # Wrap as some ModelConfig__init__ method, logger object
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
    for cfg in configurations:
        if cfg.name.split('_')[0][:2]=='RN':
            nets.append(cfg.architecture(activation=cfg.activation, num_blocks=cfg.num_blocks, num_classes=NUM_CLASSES).to(DEVICE))
        else:
            nets.append(cfg.architecture(activation=cfg.activation, num_classes=NUM_CLASSES).to(DEVICE))
        if cfg.name.split('_')[1]=="SGDwM":
            opts.append(cfg.optimizer(nets[-1].parameters(), lr=cfg.lr, momentum=MOMENTUM, weight_decay=cfg.weight_decay))
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
        cprint(f'Epoch: {epoch:>3}', color='blue')
        ### Train steps
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
        
        ### Test steps
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
                
            ### Primitive logging
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
                   
        log2png(
            logs_names=[cfg.name for cfg in configurations],
            logs_tstep=tstep_log,
            logs_tacc=tacc_log,
            logs_tloss=tloss_log,
            logs_vstep=vstep_log,
            logs_vacc=vacc_log,
            logs_vloss=vloss_log,
            max_epoch=MAX_EPOCHS,
        )
            
if __name__=="__main__":
    main()