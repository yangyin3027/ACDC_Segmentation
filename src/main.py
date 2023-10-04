from datasets import *
from network import *
from utils import *
from losses import *

import torch
from torch import nn
import numpy as np

import os
import time

import argparse
from functools import partial

logger = get_logger('Trainer')

saved_dir = 'tmp/exp12'

class_weight = torch.Tensor([0.1581, 46, 36, 26])
class_weight = class_weight / class_weight.sum()

os.makedirs(saved_dir, exist_ok=True)
csvlog = CSVLogger(saved_dir + '/checkpoints.csv', append=True)
csvlog('epoch', 'training_loss', 'val_loss', 
      'training_dice_bg', 'train_dice_rv', 'train_dice_myo', 'train_dice_lv',
      'val_dice_bg', 'val_dice_rv', 'val_dice_myo', 'val_dice_lv')

def initialize_model(model, args):
    model.to(args.device)
    if args.loss == 'dice':
        criterion = DiceLoss(weight=class_weight, softmax=True).to(args.device)
    elif args.loss == 'ce':
        criterion = nn.CrossEntropyLoss(weight=class_weight).to(args.device)
    elif args.loss == 'dicece':
        criterion = 0.5 * (nn.CrossEntropyLoss(weight=class_weight).to(args.device) +
                           DiceLoss(weight=class_weight, softmax=True).to(args.device))
    else:
        raise NotImplementedError('only loss types: dice, bce, ce, dicece and dicebce, are implemented')
    
    if args.optim_type == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    elif args.optim_type=='adam':
        optimizer = torch.optim.Adam(model.parameters(),
                                    lr=args.lr,
                                    weight_decay=args.weight_decay)
    else:
        NotImplementedError('optimizer type not implemented!')
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                            mode='min',
                                                            patience=5,
                                                            min_lr=1e-6,
                                                            factor=0.1)
    return model, criterion, optimizer, scheduler
        
def initialize_dataloader(args):    
    train_kwargs = dict(base_dir=args.data,
                        batch_size=args.batch_size,
                        size=args.img_size)
    if args.device == torch.device('cuda'):
        cuda_kwargs = {'num_workers': 1,
                    'pin_memory': True,
                    'shuffle': True}
        train_kwargs.update(cuda_kwargs)
    
    train_loader = dataloader(split='train',
                                **train_kwargs)
    val_loader = dataloader(split='val',
                            **train_kwargs)
    
    return train_loader, val_loader

def train_one_epoch(model, trainloader, criterion, optimizer, epoch):

    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ":6.3f")
    losses = AverageMeter('Loss', ':.4e')
    bg_dice = AverageMeter('DiceScore_bg', ':6.3f')
    rv_dice = AverageMeter('DiceScore_rv', ':6.3f')
    myo_dice = AverageMeter('DiceScore_myo', ':6.3f')
    lv_dice = AverageMeter('DiceScore_lv', ':6.3f')
    progress = ProgressMeter(
        len(trainloader),
        [batch_time, data_time, losses, bg_dice, rv_dice, myo_dice,lv_dice],
        prefix='Epoch: [{}]'.format(epoch),
    )

    model.train()
    end = time.time()
    for i, (images, labels) in enumerate(trainloader):
        # measure data loading time
        data_time.update(time.time() - end)
        
        # move data to right devices in case of distributed or single machine gpu/mps/cpu
        if args.device == torch.device("cuda"):
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

        # compute gradient to do parameter update
        loss = criterion(output, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.update(loss.item())

        with torch.no_grad():
            dice = dice_coeff(output, labels, weight=class_weight)    

        bg_dice.update(dice[0].item())
        rv_dice.update(dice[1].item())
        myo_dice.update(dice[2].item())
        lv_dice.update(dice[3].item())

        batch_time.update(time.time() - end)
        end = time.time()

        if i % 10 == 0:
            progress.display(i)

    res = dict(epoch=epoch, 
                loss=losses.avg,
                bg = bg_dice.avg,
                rv = rv_dice.avg,
                myo = myo_dice.avg,
                lv = lv_dice.avg)
    return res
    

def validate(model, val_dataloader, criterion, epoch):

    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    bg_dice = AverageMeter('DiceScore_bg', ':6.3f')
    rv_dice = AverageMeter('DiceScore_rv', ':6.3f')
    myo_dice = AverageMeter('DiceScore_myo', ':6.3f')
    lv_dice = AverageMeter('DiceScore_lv', ':6.3f')
    progress = ProgressMeter(
        len(val_dataloader),
        [batch_time, losses, bg_dice, rv_dice, myo_dice,lv_dice ],
        prefix='Epoch: [{}]'.format(epoch),
    )

    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, labels) in enumerate(val_dataloader):

            # move data to right devices in case of distributed or single machine gpu/mps/cpu
            if args.device == torch.device('cuda'):
                images = images.cuda(args.device, non_blocking=True)
                labels = labels.cuda(args.device, non_blocking=True)
            else:
                images = images.to(args.device)
                labels = labels.to(args.device)
            
            # compute ouput
            output = model(images)

            loss = criterion(output, labels)

            dice = dice_coeff(output, labels, weight=class_weight)  
            losses.update(loss.item())
            # ious.update(iou.item())
            bg_dice.update(dice[0].item())
            rv_dice.update(dice[1].item())
            myo_dice.update(dice[2].item())
            lv_dice.update(dice[3].item())

            batch_time.update(time.time() - end)
            end = time.time()

            if i % 10 == 0:
                progress.display(i)
    
    res = dict(epoch=epoch, 
                loss=losses.avg,
                bg = bg_dice.avg,
                rv = rv_dice.avg,
                myo = myo_dice.avg,
                lv = lv_dice.avg)
    return res
        
def main(model, args):

    ##-----------------------------initialize dataloader ------------------------------##
    train_loader, val_loader = initialize_dataloader(args)
    
    logger.info(f"data loaded with {len(train_loader.dataset)} training data and {len(val_loader.dataset)} val data")

    ##-----------------------------initialize model------------------------------------##
    model, criterion, optimizer, scheduler = initialize_model(model, args)

    logger.info(f"{args.device} device is used for training")

    best_score = float('inf')
    iteration = 0
    for epoch in range(args.epochs):

        train_res = train_one_epoch(model, train_loader, criterion, optimizer, epoch)

        
        val_res = validate(model, val_loader, criterion, epoch)
        
        if val_res['loss'] < best_score:
            best_score = val_res['loss']

            torch.save(model.state_dict(), args.checkpoint_file)
            logger.info(f'Best model saved @ {args.checkpoint_file} with loss @{best_score}')
            iteration = 0
        elif iteration <= args.maximum_iteration:
            iteration += 1
        else:
            logger.info(f'Training stopped by reaching {args.maximum_iteration} iterations with val_loss@ {best_score}')
            break
        
        scheduler.step(val_res['loss'])
        logger.info(f"current learning rate @{optimizer.param_groups[0]['lr']}")
        
        csvlog(f'{epoch}', f"{train_res['loss']}", f"{val_res['loss']}", 
              f"{train_res['bg']}", f"{train_res['rv']}", f"{train_res['myo']}", f"{train_res['lv']}",
              f"{val_res['bg']}", f"{val_res['rv']}", f"{val_res['myo']}", f"{val_res['lv']}")
        
        if args.dry_run:
            logger.info('Training stopped after 1 epoch sanity check')
            break
 

def parser_args():
    parser = argparse.ArgumentParser()

    ##-------------------datatset related-------------------------------##
    parser.add_argument('--data', type=str, default='~/scratch/data/slices', metavar='DIR',
                        help='path to dataset')
    parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                        help='input batch size for training (default: 16)')
    parser.add_argument('--test-size', type=int, default=100, metavar='N',
                        help='input batch size for testing (default: 100)')
    parser.add_argument('--n_classes', type=int, default=4, metavar='CLS',
                        help='number of classes for the model (default: 4)')
    parser.add_argument('--img-channels', type=int, default=1, metavar='C',
                        help='number of channels in input images (default: 1)')
    parser.add_argument('--img-size', type=int, default=224, metavar='S',
                        help='img size for model to train (default: 224)')
    
    ##---------------------epochs--------------------------------------##
    parser.add_argument('--epochs', type=int, default=1000, metavar='N',
                        help='number of epochs for train (default: 100)')
    
    ##----------------------learning rate------------------------------##
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=1e-4, metavar='W',
                        help='adam optimizer weight decay (default: 1e-4)')
    parser.add_argument('--init', type=str, default='normal',
                        help='initialize method for method parameters')
    
    parser.add_argument('--loss', type=str, default='dice',
                        help='loss function used for training (default: dice)')
    
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
    parser.add_argument('--rank', type=int, default=-1,
                        help='node rank for distributed training')
    parser.add_argument('--dis-url', type=str, default='env://',
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', type=str, default='nccl',
                        help='distributed backend')
    parser.add_argument('--local-rank', type=int, default=-1,
                        help='local rank for distributed training')
    
    ##----------------------save checkpoint-----------------------------##
    parser.add_argument('--checkpoint-file', type=str, default=saved_dir + '/checkpoint.pth.tar',
                        help='checkpoint file path, to load and save to')
    parser.add_argument('--maximum-iteration', type=int, default=20,
                        help='maximum allowed iterations for early stopping')
                        
    parser.add_argument('--model-type', type=str, default='attenunet',
                        help='type of models to use')
    parser.add_argument('--optim-type', type=str, default='sgd',
                        help='type of optimizer')
                        
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    start = time.time()
    cur_time = time.localtime(start)
    logger.info(f'Training started @ {time.asctime(cur_time)}')
    args = parser_args()

    if args.cpu:
        args.device = torch.device('cpu')
    elif torch.cuda.is_available():
        args.device = torch.device('cuda')
        torch.backends.cudnn.benchmark = True
    elif torch.backends.mps.is_available():
        args.device = torch.device('mps')
    else:
        args.device = torch.device('cpu')

    if args.model_type == 'unet':
        model = UNet(args.n_classes, args.img_channels)
    elif args.model_type =='attenunet':
        model = AttenUnet(args.n_classes, args.img_channels)
    else:
        NotImplementedError('model type not implemented!')

    init = partial(init_weights, init_type=args.init)
    model.apply(init)
    logger.info(f'model weights initialized by {args.init}')

    main(model, args)

    logger.info(f'Training completed after {(time.time()-start)/60:2f} minutes')