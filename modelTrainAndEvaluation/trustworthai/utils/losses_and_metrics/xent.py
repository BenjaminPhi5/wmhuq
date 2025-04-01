import torch
import torch.nn as nn

def xent_loss(weight, reduction):
    if reduction == "mean_sum":
        return mean_sum_xent_loss(weight)
    else:
        return torch.nn.CrossEntropyLoss(weight=weight, reduction=reduction)


class mean_sum_xent_loss(nn.Module):
    def __init__(self, weight=None):
        super().__init__()
        self.weight = weight
        
    def forward(self, pred, target):
        weight = self.weight
        if weight != None:
            weight=weight.to(pred.device)
        l = torch.nn.functional.cross_entropy(pred, target, weight=weight, reduction='none')
        bs = l.shape[0]
        l = l.view(bs, -1)
        # sum per pixel and then take mean over them all.
        return l.sum(dim=1).mean()


class dice_xent_loss(nn.Module):
    def __init__(self, dice_fn, xent_fn, dice_weight, xent_weight):
        super().__init__()
        self.dice_fn = dice_fn
        self.xent_fn = xent_fn
        self.dice_weight = dice_weight
        self.xent_weight = xent_weight
    
    def forward(self, pred, target):
        return (self.dice_fn(pred, target) * self.dice_weight) + (self.xent_fn(pred, target) * self.xent_weight)