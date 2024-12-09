import os
import tqdm
from termcolor import cprint
from copy import deepcopy

from __train_loop_functions import accuracy, get_n_params

import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import torch
from torch import nn

from configs.configRRf4 import get_configsRRf4

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
    './ckpt_final/RCog_199.pt',
    './ckpt_final/RC20_199.pt',
]

configurations = get_configsRRf4()[:2] ############################################

[
    print(cfg.name)
    for cfg in configurations
]

###############################################################################

def main():
    
    MAX_EPOCHS = 1
    BATCH_SIZE = 128
    NUM_WORKERS = 0
    DOWNLOAD = True

    HEAVY_REGULARIZATION = True

    LINX = True
    LINY = True
    LOAD = True
    SAVE = False
    
    CUDNN = True
    if CUDNN:
        import torch.backends.cudnn as cudnn
        cudnn.benchmark = True

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./data', train=True, transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ]), download=True),
        batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True)

    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=128, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True)
    
    batches_train = len(train_loader)
    batches_test = len(test_loader)
    NUM_CLASSES = 10
    
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
    for it, cfg in enumerate(configurations):
        # print(cfg.name.split('_')[0][:4], type(cfg.architecture))
        load_path = LOAD_PATHS[it]
        if LOAD:
            checkpoint = torch.load(load_path, weights_only=True, map_location=torch.device(DEVICE))
        if cfg.name.split('_')[0][:2]=='RN' or cfg.name.split('_')[0][:2]=='RR':
            nets.append(cfg.architecture(activation=cfg.activation, num_blocks=cfg.num_blocks, num_classes=NUM_CLASSES).to(DEVICE))
            if LOAD:
                nets[-1].load_state_dict(checkpoint['model_state_dict'])
        elif cfg.name.split('_')[0][:4]=='RCog':
            nets.append(cfg.architecture().to(DEVICE))
            if LOAD:
                nets[-1].load_state_dict(checkpoint['model_state_dict'])    
        else: # did not treat this case
            nets.append(cfg.architecture(activation=cfg.activation, num_classes=NUM_CLASSES).to(DEVICE))    
            if LOAD:
                nets[-1].load_state_dict(checkpoint['model_state_dict'])
        

        if cfg.name.split('_')[1]=="SGDwM": # did not treat this case
            opts.append(cfg.optimizer(nets[-1].parameters(), lr=cfg.lr, momentum=MOMENTUM, weight_decay=cfg.weight_decay))
            if LOAD:    
                opts[-1].load_state_dict(checkpoint['optimizer_state_dict'])
            schs.append(scheduler(opts[-1]))
        else:
            opts.append(cfg.optimizer(nets[-1].parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay))
            # schs.append(scheduler(opts[-1]))
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
        cprint(f'Evaluating', color='blue')

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
        
        for cfg, test_acc in zip(configurations, vacc_log):
            print(f'{cfg.name:<40}{test_acc[0]:>10}')          
            
if __name__=="__main__":
    main()