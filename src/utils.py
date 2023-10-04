import logging
import sys
import os

from typing import List, Tuple

import torch

loggers = {}

def get_logger(name='main', level=logging.INFO):
    global loggers
    if loggers.get(name) is not None:
        return loggers[name]
    else:
        logger = logging.getLogger(name)
        logger.setLevel(level)

        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            '%(asctime)s | [%(threadName)s] | %(levelname)s | %(name)s -> %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        loggers[name] = logger
        return logger

class AverageMeter(object):
    '''Compute and stores the average and current values'''
    def __init__(self, name:str, fmt:str=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()
    
    def reset(self) -> None:
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
    def __str__(self):
        fmtstr = "{name}: {avg" + self.fmt + "} "
        return fmtstr.format(**self.__dict__)

class ProgressMeter(object):
    def __init__(self, num_batches:int, meters:List[AverageMeter], prefix:str=''):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
    
    def display(self, batch:int) -> None:
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print(' | '.join(entries))
    
    def _get_batch_fmtstr(self, num_batches:int) -> str:
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"

class CSVLogger(object):
    def __init__(self, fname, append=False, sep=','):
        self.fname = fname
        if append:
            self.mode = 'a'
        else:
            self.mode = 'w'
        self.sep = sep
    
    def __call__(self, *args):
        f = open(self.fname, self.mode)
        context = self.sep.join([a for a in args])
        f.write(context + '\n')
        f.close()

def learnable_params_summary(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def load_checkpoint(path, model, optimizer, from_ddp=False):
    '''Load previous checkpoints to resume training

    Args:
        path (str): path to checkpoint
        model (torch model): model to load pretrained state_dict to
        optimizer (torch optimizer): torch optimizer to load optimizer state_dict to
        from_ddp (bool, optional): load DistributedDataParallel checkpoint to regular model.
    
    Returns:
        model, optimizer, epoch_num, loss
    '''
    # load checkpoint
    checkpoint = torch.load(path)

    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    loss = checkpoint['loss']
    return model, optimizer, checkpoint['epoch'], loss,

def save_checkpoint(epoch, model, optimizer, loss, path):
    '''Save model as checkpoint'''
    torch.save({
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)

def optimizer_to_gpu(optim, device):
    for param in optim.state.values():
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)

def compute_class_weights(dataloader):
    '''Adapted from sklearn.utils.weights.compute_class_weights
    using n_samples/(n_classes * np.bincount(y))
    '''
    class_weights = []
    for _, labels in dataloader:
        classes, freqs = torch.unique(labels, return_counts=True)
        weights = labels.numel()/(classes.numel() * freqs)
        class_weights.append(weights[classes])
    
    class_weights = torch.stack(class_weights)
    return class_weights.mean(0)

def compute_mean_std(dataloader, img_channel=1):
    '''compute mean and std for batch of images 
    for data normalization
    '''
    psum = torch.zeros(img_channel, dtype=torch.float32)
    psum_sq = torch.zeros(img_channel, dtype=torch.float32)
    counts = 0.0

    for data, _ in dataloader:
        psum += data.sum(dim=[0, 2, 3])
        psum_sq += (data**2).sum(dim=[0, 2, 3])
        counts += data.size(0) * data.size(2) * data.size(3)
    
    mean = psum / counts
    var = psum_sq / counts - mean**2
    std = torch.sqrt(var)
    return mean, std

