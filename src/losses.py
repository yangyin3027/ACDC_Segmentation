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
    # alternative approach
    # shape = np.array(target.shape)
    # shape[1] = num_classes
    # shape = tuple(shape)
    # on_hot = torch.zeros(shape, device=target.device,
    #                      requires_grad=True).scatter_(1, target, 1)
    
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
        and multiclass segmentation
        Calculate loss and metrics directly on logits is better
        than argmax    
    '''
    assert preds.ndim >= 2, 'pred dimension is smaller than 2'
    if softmax:
        preds = torch.nn.functional.softmax(preds, dim=1)
        num_classes = preds.size(1)
        # permute preds to make class dim the last
        preds = preds.permute(0,2,3,1)
    else:
        num_classes = -1
    
    # if target is not one_hot_encoded, convert it
    # if torch.max(targets) > 1 or targets.ndim < 4:
    targets = one_hot_encoder(targets, num_classes=num_classes)
    
    # permute encoded last dimension to be the channel dimension
    # use reshape instead of view, as it's not continous dim after permute
    preds = preds.reshape(-1, num_classes)
    targets = targets.reshape(-1, num_classes)

    # calculate dice with class dimensions
    # calculate logits by **2
    y_pred_sum = torch.sum(preds**2, dim=0)
    y_true_sum = torch.sum(targets**2, dim=0)

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

class DiceCELoss(nn.Module):
    '''Combine dice and crossentropy loss'''
    def __init__(self, weight=None, softmax=True, eps=1e-6, **kwargs):
        super(DiceCELoss, self).__init__()
        self.dice = DiceLoss(weight=weight, softmax=softmax, eps=eps)
        self.ce = nn.CrossEntropyLoss(weight=weight, **kwargs)
    
    def forward(self, preds, targets):
        loss = self.dice(preds, targets) * 0.2 + self.ce(preds, targets)
        return loss

class CrossEntropyRegularized(nn.Module):
    '''CrossEntropy loss with information entropy as a regularization'''
    def __init__(self, eps=0.5, reduction='mean'):
        super().__init__()
        self.eps = eps
        self.nll = nn.NLLLoss(reduction=reduction)
        self.softmax = nn.Softmax(1)
    
    def update_eps(self):
        self.eps = self.eps * 0.1
    
    def forward(self, preds, targets):
        '''
        Args:
            preds: unnormalized probabilities
            targets: [N, C-1] index tensor
        '''
        preds = self.softmax(preds)
        ce = self.nll(preds.log(), targets)
        reg = preds * preds.log()
        reg = reg.sum(1).mean()
        loss_total = ce + reg * self.eps
        return loss_total

class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.0, 
                 dim=-1, weight=None):
        super().__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.weight = weight
        self.cls = classes
        self.dim = dim
    
    def forward(self, preds, targets):
        assert 0 <= self.smoothing < 1, 'smoothing can only be in [0, 1]'
        preds = preds.log_softmax(dim=self.dim)

        if self.weight is not None:
            self.weight = self.weight.unsqueeze(0)
            self.weight = self.weight.to(preds.device)
            preds = preds * self.weight
        
        with torch.no_grad():
            true_dist = torch.zeros_like(preds)
            true_dist.fill_(self.smoothing / (self.cls-1))
            true_dist.scatter_(1, targets.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist*preds, dim=self.dim))


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