from datasets.dataloader import *
from network import *
from utils import *
from losses import *

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
import numpy as np

import os
import time
import random
from tqdm import tqdm

import argparse
from functools import partial

saved_dir = './tmp/Atten_synapse'
os.makedirs(saved_dir, exist_ok=True)

logger = get_logger('Trainer')
# for Synpase, precomputed
class_weight = torch.Tensor([0.15, 91.2, 47.6, 7.5, 32.1, 110.1, 183.2])
# class_weight = torch.Tensor([0.1, 0.3, 0.3, 0.3])
class_weight = class_weight / class_weight.sum()
# pos_weight = torch.Tensor([0.01, 250., 200., 150.])


csvlog = CSVLogger(saved_dir + '/checkpoints.csv', append=True)
csvlog('epoch', 'training_loss', 'val_loss', "lr",
      'training_dice_score', 'val_dice_score',)

def initialize_model(args):
    if args.model == 'attenunet':
        model = AttenUnet(args.num_classes, args.img_channels)
    elif args.model == 'unet':
        model = UNet(args.num_classes, args.img_channels)
    init = partial(init_weights, init_type=args.init)
    model.apply(init)
    
    if args.local_rank <=0:
        logger.info(f'model weights initialized by {args.init}')

    model.to(args.device)

    criterion = DiceCELoss(weight=class_weight).to(args.device)
    # criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight).to(args.device)
    
    if args.distributed:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])
      
    optimizer = torch.optim.Adam(model.parameters(),
                                lr=args.lr,
                                weight_decay=args.weight_decay)   
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15,T_mult=1,eta_min=1e-6,last_epoch=-1)
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max')
    warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: min(float(epoch+1)/20, 1.))
    
    return model, criterion, optimizer, scheduler, warmup_scheduler
        
def initialize_dataloader(args):    
    train_kwargs = dict(root=args.data,
                        batch_size=args.batch_size,
                        output_size=args.img_size,
                        dds=False,)
    if args.distributed:
        cuda_kwargs = {'num_workers': 1,
                    'pin_memory': True,
                    'dds': True}
        train_kwargs.update(cuda_kwargs)

    else:
        if args.device == torch.device('cuda'):
            cuda_kwargs = {'num_workers': 1,
                        'pin_memory': True,}
            train_kwargs.update(cuda_kwargs)
    
        train_loader, val_loader = get_dataloader(**train_kwargs)
    if args.local_rank <=0:
        logger.info(f"data loaded with {len(train_loader.dataset)} training data and {len(val_loader.dataset)} val data")   
    return train_loader, val_loader

def train_one_epoch(model, train_loader, criterion, optimizer, epoch, args):

    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ":6.3f")
    losses = AverageMeter('Loss', ':.4e')
    dice_score = AverageMeter('DiceScore', ':6.3f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, dice_score],
        prefix='Epoch: [{}]'.format(epoch),
    )

    model.train()
    end = time.time()
    with tqdm(train_loader, desc=f'Training epoch {epoch} ->',
              total = len(train_loader)) as pbar:
        for i, (images, labels) in enumerate(pbar):
            # measure data loading time
            data_time.update(time.time() - end)
            
            # move data to right devices in case of distributed or single machine gpu/mps/cpu
            if args.device == torch.device('cuda:0'):
                images = images.cuda(args.device, non_blocking=True)
                labels = labels.cuda(args.device, non_blocking=True)
            else:
                images = images.to(args.device)
                labels = labels.to(args.device)

            # compute ouput
            # output is just unnormalized logits
            # CE will just use it as is
            # Diceloss will add softmax, argmax and one-hot-encoding
            output = model(images)

            if args.model == 'FCT':
                down2 = F.interpolate(labels.unsqueeze(1).float(), scale_factor=1/2)
                down3 = F.interpolate(labels.unsqueeze(1).float(), scale_factor=1/4)

                loss = (criterion(output[2], labels.long()) * 0.57 
                        + criterion(output[1], down2.squeeze().long()) * 0.29 
                        + criterion(output[0], down3.squeeze().long()) * 0.14)
            # bce_pred = F.sigmoid(output)
            # bce_pred = bce_pred.permute(0, 2, 3, 1)
            
            # bce_labels = F.one_hot(labels, num_classes=args.n_classes)
            # bce_labels = bce_labels.float()
            else:
                loss = criterion(output, labels.long())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                if args.model == 'FCT':
                    output = output[2]
                dice = dice_coeff(output, labels)    
            
            if args.distributed:
                dist.barrier()
            if args.local_rank <= 0:
                losses.update(loss.item())    
                dice_score.update(torch.mean(dice).item())

                batch_time.update(time.time() - end)
                end = time.time()

                pbar.set_postfix(time=batch_time.avg,
                                loss=losses.avg,
                                 dice_score=dice_score.avg)
                # if i % 10 == 0:
                #    progress.display(i)

    res = dict(epoch=epoch, 
                loss=losses.avg,
                dice_score=dice_score)
    return res
    

def validate(model, val_loader, criterion, epoch, args):

    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    dice_score = AverageMeter('DiceScore', ':6.3f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, dice_score],
        prefix='Epoch: [{}]'.format(epoch),
    )

    model.eval()

    with torch.no_grad():
        end = time.time()
        with tqdm(val_loader, desc=f'Validation epoch {epoch} ->',
                  total=len(val_loader)) as pbar:
            for i, (images, labels) in enumerate(val_loader):

                # move data to right devices in case of distributed or single machine gpu/mps/cpu
                if args.device == torch.device('cuda:0'):
                    images = images.cuda(args.device, non_blocking=True)
                    labels = labels.cuda(args.device, non_blocking=True)
                else:
                    images = images.to(args.device)
                    labels = labels.to(args.device)
                
                # compute ouput
                output = model(images)
                # bce_pred = F.sigmoid(output)
                # bce_pred = bce_pred.permute(0, 2, 3, 1)
                # bce_labels = F.one_hot(labels, num_classes=args.n_classes)
                # bce_labels = bce_labels.float()
                
                if args.model == 'FCT':
                    output = output[2]
                loss = criterion(output, labels.long())
                dice = dice_coeff(output, labels) 
                
                if args.distributed:
                    dist.barrier() 
                if args.local_rank <= 0:
                    losses.update(loss.item())
                    # ious.update(iou.item())
                    dice_score.update(torch.mean(dice).item())

                    batch_time.update(time.time() - end)
                    end = time.time()

                    pbar.set_postfix(time=batch_time.avg,
                                loss=losses.avg,
                                 dice_score=dice_score.avg)

                    # if i % 10 == 0:
                    #    progress.display(i)
    
    res = dict(epoch=epoch, 
                loss=losses.avg,
                dice_score=dice_score)
    return res

def save_checkpoints(args, model,best_score):
    if args.local_rank == 0:
        ckp = model.module.state_dict()
        torch.save(ckp, args.checkpoint_file)
    elif args.local_rank < 0:
        ckp = model.state_dict()
        torch.save(ckp, args.checkpoint_file)
    logger.info(f'Best model saved @ {args.checkpoint_file} with loss @ {best_score}')

def train(args, model, train_loader, val_loader, optimizer, criterion, scheduler, warmup_scheduler):

    best_score = float('inf')
    iteration = 0
    flag = torch.zeros(1).to(args.device)

    with tqdm(range(args.epochs), desc='Training ->>',
              total=args.epochs) as pbar:
        for epoch in pbar:
        
            if args.distributed:
                train_loader.sampler.set_epoch(epoch)
                val_loader.sampler.set_epoch(epoch)
                
            ##----------------------------------train----------------------------------------##
            train_res = train_one_epoch(model, train_loader, criterion, optimizer, epoch, args)
            
            ##----------------------------------validate-------------------------------------##
            val_res = validate(model, val_loader, criterion, epoch, args)

            if args.distributed:
                dist.barrier()

            if epoch <= 20:
                warmup_scheduler.step()
            else:
                scheduler.step()
            
            lr = optimizer.param_groups[0]['lr']
            
            if args.local_rank <= 0:

                pbar.set_postfix(lr=f'{lr}',
                                 train_loss=train_res['loss'],
                                 train_dice=train_res['dice_score'],
                                 val_loss=val_res['loss'],
                                 val_dice=val_res['dice_score'])

                logger.info(f"current learning rate @{lr}")  

                csvlog(f'{epoch}', f"{train_res['loss']}", f"{val_res['loss']}", f"{lr}",
                f"{train_res['dice_score']}", f"{val_res['dice_score']}") 
                
                if val_res['loss'] < best_score:
                    best_score = val_res['loss']
                    save_checkpoints(args, model, best_score)
                    iteration = 0
                elif iteration <= args.maximum_iteration:
                    iteration += 1
                else:
                    flag +=1
                    logger.info(f'Training stopped by reaching {args.maximum_iteration} iterations with loss@ {best_score}')
            
            if args.distributed:
                dist.all_reduce(flag, op=dist.ReduceOp.SUM)
            if flag == 1:
                break
                                    
            if args.dry_run:
                logger.info('Training stopped after 1 epoch sanity check')
                break             

def parser_args():
    parser = argparse.ArgumentParser()
    ##-------------------datatset related-------------------------------##
    parser.add_argument('--data', type=str, default='~/scratch/data/slices', metavar='DIR',
                        help='path to dataset')
    parser.add_argument('--batch-size', type=int, default=24, metavar='N',
                        help='input batch size for training (default: 16)')
    parser.add_argument('--num-classes', type=int, default=4, metavar='CLS',
                        help='number of classes for the model (default: 4)')
    parser.add_argument('--img-channels', type=int, default=1, metavar='C',
                        help='number of channels in input images (default: 1)')
    parser.add_argument('--img-size', type=int, default=256, metavar='S',
                        help='img size for model to train (default: 224)')
    
    ##---------------------epochs--------------------------------------##
    parser.add_argument('--epochs', type=int, default=1000, metavar='N',
                        help='number of epochs for train (default: 100)')
        
    ##----------------------learning rate------------------------------##
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--weight-decay', type=float, default=1e-4, metavar='W',
                        help='adam optimizer weight decay (default: 1e-4)')
    parser.add_argument('--init', type=str, default='kaiming',
                        help='initialize method for method parameters')
    parser.add_argument('--model', type=str, default='attenunet',
                        help='model architecture to use (default: attenunet)')
    
    ##----------------------single machine training related-------------##
    parser.add_argument('--cpu', action='store_true', default=False,
                        help='use cpu for training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quick sanity check')
    parser.add_argument('--seed', type=int, default=2024, metavar='S',
                        help='random seed (default: 2023)')
    
    ##-----------------------for distributed training-------------------##
    parser.add_argument('--world-size', type=int, default=-1,
                        help='number of nodes for distributed training')
    parser.add_argument('--local-rank', type=int, default=-1,
                        help='necessary for torch.distributed.launch')
    parser.add_argument('--dist-url', type=str, default='env://',
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', type=str, default='nccl',
                        help='distributed backend')
    
    ##----------------------save checkpoint-----------------------------##
    parser.add_argument('--checkpoint-file', type=str, default=saved_dir + '/checkpoint.pth.tar',
                        help='checkpoint file path, to load and save to')
    parser.add_argument('--maximum-iteration', type=int, default=15,
                        help='maximum allowed iterations for early stopping')
                                        
    args = parser.parse_args()
    return args

def set_random_seed(args):
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)
    random.seed(args.seed)
    
def dist_setup(args):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '21532'

    torch.cuda.set_device(args.device)
    dist.init_process_group(args.dist_backend,
                            rank=args.local_rank,
                            world_size=args.world_size,
                            init_method=args.dist_url)

def main():
    args = parser_args()
    set_random_seed(args)

    if 'WORLD_SIZE' in os.environ:
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.local_rank = int(os.environ['LOCAL_RANK'])
    args.distributed = args.world_size > 1

    if args.distributed:
        args.device = torch.device('cuda:{}'.format(args.local_rank))
        dist_setup(args)
     
        if args.local_rank <=0:
            logger.info('Distributed Data Parallel Training was initiated with {} gpus'.format(args.world_size))
    else:
        if args.cpu:
            args.device = torch.device('cpu')
        elif torch.cuda.is_available():
            args.device = torch.device('cuda:0')
        elif torch.backends.mps.is_available():
            args.device = torch.device('mps')
        else:
            args.device = torch.device('cpu')
        
        logger.info('Single machine training was performed on {} devices'.format(args.device))

    if args.local_rank <=0:
        start = time.time()
        cur_time = time.localtime(start)
        logger.info(f'Training started @ {time.asctime(cur_time)}')

    model, criterion, optimizer, scheduler, warmup_scheduler = initialize_model(args)
    train_loader, val_loader = initialize_dataloader(args)
    
    train(args, model, train_loader, val_loader, optimizer, criterion, scheduler, warmup_scheduler)
    
    if args.local_rank <=0:
        logger.info(f'Training completed after {(time.time()-start)/60:.1f} minutes')
        
    if args.distributed:
        torch.cuda.empty_cache()
        dist.destroy_process_group()

if __name__ == '__main__':
    
    main()