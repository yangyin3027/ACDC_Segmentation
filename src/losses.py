import torch
from torch import nn

def one_hot_encoder(target: torch.tensor, num_classes=-1) -> torch.tensor:

    # make sure target is an index tensor
    if torch.is_floating_point(target):
        target = target.long()

    # if num_classes is not specified, use the maxValue + 1
    if num_classes < 0:
        num_classes = torch.max(target) + 1
        num_classes = num_classes.item()
    
    one_hot = torch.eye(num_classes,
                        device=target.device,
                        requires_grad=True)
    
    # last dimension is the encoded
    target_one_hot = one_hot[target] 
    return target_one_hot

###################################################################
##                      Metrics                                  ##
###################################################################
def dice_coeff(preds, targets, 
               weight=None, 
               softmax=True,
               eps=1e-6):
    '''Dice coefficient supported for both binary segmentation
        and multiclass segmentation'''
    assert preds.ndim >= 2, 'pred dimension is smaller than 2'
    if softmax:
        preds = torch.nn.functional.softmax(preds, dim=1)
        num_classes = preds.size(1)
        preds = torch.argmax(preds, dim=1)
    else:
        num_classes = -1
    # one-hot encoding for both preds and targets required
    # dim 1 contains num of classes
    
    preds = one_hot_encoder(preds, num_classes=num_classes)

    # if target is not one_hot_encoded, convert it
    # if torch.max(targets) > 1 or targets.ndim < 4:
    targets = one_hot_encoder(targets, num_classes=num_classes)
    
    # permute encoded last dimension to be the channel dimension

    preds = preds.view(-1, num_classes)
    targets = targets.view(-1, num_classes)

    # calculate dice with class dimensions
    y_pred_sum = torch.sum(preds, dim=0)
    y_true_sum = torch.sum(targets, dim=0)

    intersection = torch.sum(preds * targets, dim=0)
    union = y_pred_sum + y_true_sum

    dice = (2*intersection + eps) / (union + eps)

    if weight is not None:
        weight = weight.to(dice.device)
        dice = weight * dice
    
    # for binary cases
    if len(dice) == 1:
        dice = dice.squeeze()
    return dice

def iou_scores(preds, targets, 
               weight=None, 
               softmax=True,
               eps=1e-6):
    '''Ious scores supported for both binary segmentation
     and multiclass segmentation'''
    
    assert preds.ndim >= 2, 'pred dimension is smaller than 2'
    if softmax:
        preds = torch.nn.functional.softmax(preds, dim=1)
        num_classes = preds.size(1)
        preds = torch.argmax(preds, dim=1)
    else:
        num_classes = -1

    preds = one_hot_encoder(preds, num_classes=num_classes)

    # if target is not one_hot_encoded, convert it
    if torch.max(targets) > 1 or targets.ndim < 4:
        targets = one_hot_encoder(targets, num_classes=num_classes)
    
    # permute encoded last dimension to be the channel dimension
    if preds.ndim == 4:
        preds = preds.permute(0, 3, 1, 2)
        targets = targets.permute(0, 3, 1, 2)
    
    preds = preds.view(-1, num_classes)
    targets = targets.view(-1, num_classes)

    # calculate dice with class dimensions
    y_pred_sum = torch.sum(preds, dim=0)
    y_true_sum = torch.sum(targets, dim=0)

    intersection = torch.sum(preds * targets, dim=0)
    union = y_pred_sum + y_true_sum

    ious = (intersection + eps) / (union - intersection + eps)

    if weight is not None:
        weight = weight.to(ious.device)
        ious = weight * ious
    
    # for binary cases
    if len(ious) == 1:
        ious = ious.squeeze()

    return ious

###################################################################
##                      Losses                                   ##
###################################################################
class DiceLoss(nn.Module):
    '''Dice loss for binary segmentation'''
    def __init__(self, weight=None, softmax=True, eps=1e-6):
        super(DiceLoss, self).__init__()
        self.weight = weight
        self.softmax = softmax
        self.eps = eps
    
    def forward(self, preds, targets):
        dice = dice_coeff(preds, targets, 
                          self.weight, self.softmax, self.eps)
        
        if dice.ndim != 0:
            dice = torch.sum(dice)
        return 1 - dice

class TverskyLoss(nn.Module):
    '''Loss for binary segmentation'''
    def __init__(self, alpha=0.5, beta=0.5, eps=1e-6):
        super(TverskyLoss, self).__init__()
        self.alpha = 0.5
        self.beta = 0.5
        self.eps = eps
    
    def forward(self, pred, target, softmax=False):
        assert pred.ndim >= 2, 'pred dimension is smaller than 2'
        if softmax:
            preds = torch.nn.functional.softmax(preds, dim=1)
            pred = torch.argmax(pred, dim=1)
        pred = pred.view(-1)

        if torch.is_floating_point(target):
            target = target.long()
        target = target.view(-1)

        TP = (pred * target).sum()
        FP = (pred * (1-target)).sum()
        FN = ((1-pred) * target).sum()

        tversky = (TP + self.eps) / (TP + self.alpha * FP + self.beta * FN + self.eps)
        return 1 - tversky
