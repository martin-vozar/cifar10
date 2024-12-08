import os
import torch

def train(train_loader, model, criterion, optimizer, epoch):
    """
        Run one train epoch
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    for i, (input, target) in enumerate(train_loader):


        target = target.cuda()
        input_var = input.cuda()
        target_var = target


        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output = output.float()
        loss = loss.float()
        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def step_train(step, x, y, nets, opts, criterion, batches_train, DEVICE, 
                  loss_v, tacc_log):
           
    for it, (net, opt) in enumerate(zip(nets, opts)):
    
        net.train()

        x = x.to(DEVICE, dtype=torch.float32) 
        y = y.to(DEVICE)               
                        
        xi = net(x)
        loss = criterion(xi, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1, norm_type=2)
        opt.step()
        
        loss_v[it] += loss.item()
        
        correct = accuracy(xi.data, y)[0]

        # measure accuracy and record loss
        prec1 = accuracy(xi.data, y)[0]
        
        tacc_log[it].append(prec1)

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res