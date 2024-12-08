import os
import tqdm
from termcolor import cprint
from copy import deepcopy

from __train_loop_functions import accuracy, get_n_params

import torch
from torch import nn

from configs.configRRf2 import get_configsRRf2
from configs.configRRf1 import get_configsRRf1
from configs.configRRf0 import get_configsRRf0
from configs.configRR4 import get_configsRR4
from configs.configRR3 import get_configsRR3
from configs.configRR2 import get_configsRR2
from configs.configRR1 import get_configsRR1
from configs.configRR0 import get_configsRR0
from configs.configRN4 import get_configsRN4
from configs.configRN3 import get_configsRN3
from configs.configRN2 import get_configsRN2
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

MOMENTUM = 0.90

MAX_EPOCHS = 200
BATCH_SIZE = 128
NUM_WORKERS = 4
DOWNLOAD = True

HEAVY_REGULARIZATION = True

LINX = True
LINY = True
LOAD = False
SAVE = True
LOAD_PATHS = [
    './ckpt_f/RR-32_SGD_lr1.0e-01_wd0.0e+00_LeakyReLU/99.pt',
    './ckpt_f/RR-32_AdamW_lr2.0e-03_wd0.0e+00_LeakyReLU/99.pt',
]

configurations = get_configsRRf2() ############################################

###############################################################################

def main():
    train_loader, test_loader, classes = get_split_dls(
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS, 
        download=DOWNLOAD, 
        heavy_regularization=HEAVY_REGULARIZATION,
    )
    batches_train = len(train_loader)
    batches_test = len(test_loader)
    NUM_CLASSES = len(classes)
    
    every_n_steps_train = len(train_loader) // 1
    if every_n_steps_train == 0:
        every_n_steps_train = 1
    
    every_n_steps_test = len(test_loader) // 1 # if ==1 => accumulated loss for test set
    if every_n_steps_test == 0:
        every_n_steps_test = 1
    
    criterion = nn.CrossEntropyLoss(reduction='mean')
    scheduler = lambda opt: torch.optim.lr_scheduler.MultiStepLR(
        optimizer=opt,
        milestones=[100, 150], 
        last_epoch= -1
    )
    
    # Wrap as some ModelConfig__init__ method, logger object
    nets = []
    opts = []
    schs = []
    loss_t = []
    loss_v = []
    tstep_log = []
    tacc_log = []
    tloss_log = []
    vstep_log = []
    vacc_log = []
    vloss_log = []
    # for cfg, load_path in zip(configurations, LOAD_PATHS):
    for cfg in configurations:
        load_path = LOAD_PATHS[0]
        if LOAD:
            checkpoint = torch.load(load_path, weights_only=True)
        if cfg.name.split('_')[0][:2]=='RN' or cfg.name.split('_')[0][:2]=='RR':
            nets.append(cfg.architecture(activation=cfg.activation, num_blocks=cfg.num_blocks, num_classes=NUM_CLASSES).to(DEVICE))
            if LOAD:
                nets[-1].load_state_dict(checkpoint['model_state_dict'])
        else: # did not treat this case
            nets.append(cfg.architecture(activation=cfg.activation, num_classes=NUM_CLASSES).to(DEVICE))

        if cfg.name.split('_')[1]=="SGDwM": # did not treat this case
            opts.append(cfg.optimizer(nets[-1].parameters(), lr=cfg.lr, momentum=MOMENTUM, weight_decay=cfg.weight_decay))
            if LOAD:    
                opts[-1].load_state_dict(checkpoint['optimizer_state_dict'])
            schs.append(scheduler(opts[-1]))
        else:
            opts.append(cfg.optimizer(nets[-1].parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay))
            schs.append(scheduler(opts[-1]))
            # opts[-1].load_state_dict(checkpoint['optimizer_state_dict']) # one optimizer is reset  

        print(f'{cfg.name:<50}{get_n_params(nets[-1]):>6}') 
            
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
        for step, (x, y) in tqdm.tqdm(enumerate(train_loader), total=batches_train):
            # print(len(nets), len(opts), len(schs))
            for it, (net, opt) in enumerate(zip(nets, opts)):
                x = x.to(DEVICE, dtype=torch.float32) 
                y = y.to(DEVICE)               
                                
                xi = net(x)
                loss = criterion(xi, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1, norm_type=2)
                opt.step()
                
                loss_t[it] += loss.item()
                
                correct[it] += accuracy(xi.data, y)[0].detach().cpu()
                
                # _, predicted = xi.max(1)
                # total[it] += y.size(0)
                # correct[it] += predicted.eq(y).sum().item()
                
            if (step+1)%every_n_steps_train==0:
                for it in range(len(nets)):
                    tstep_log[it].append(epoch+step/len(train_loader))
                    tloss_log[it].append(loss_t[it]/every_n_steps_train)
                    loss_t[it] = 0
                    tacc_log[it].append(correct[it]/batches_train/100)
                    
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
         
        for sch in schs:
            sch.step()
          
        # print(len(tstep_log), len(tstep_log[0]))
          
        # print(len(vstep_log), len(vstep_log[0])) 
        
        # print(len(tacc_log), len(tacc_log[0]))  
        
        # print(len(vacc_log), len(vacc_log[0]))  
        
        # print(len(tloss_log), len(tloss_log[0]))  
        
        # print(len(vloss_log), len(vloss_log[0]))  
                   
        log2png(
            logs_names=[cfg.name for cfg in configurations],
            logs_tstep=tstep_log,
            logs_tacc=tacc_log,
            logs_tloss=tloss_log,
            logs_vstep=vstep_log,
            logs_vacc=vacc_log,
            logs_vloss=vloss_log,
            max_epoch=MAX_EPOCHS,
            linx=LINX,
            liny=LINY,
        )
        
        if SAVE:
            for net, opt, name in zip(nets, opts, [cfg.name for cfg in configurations]):
                try:
                    _ = os.listdir(os.path.join('.', 'ckpt_f3', name))
                except:
                    os.mkdir(os.path.join('.', 'ckpt_f3', name))
                torch.save(
                    {
                    'epoch': epoch,
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': opt.state_dict(),
                    'loss': loss,
                    }, 
                    os.path.join('.', 'ckpt_f3', name, f'{epoch}.pt')
                )
            
if __name__=="__main__":
    main()