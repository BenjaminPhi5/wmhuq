import torch
from trustworthai.utils.losses_and_metrics.dice_loss import get_normalized_probs
import torch.nn as nn
from trustworthai.utils.losses_and_metrics.dice_loss import get_normalized_probs


def tversky_index(y_pred, y_true, beta, addone=False):
    s0 = y_pred.shape[0]
    y_pred = y_pred.view(s0,-1)
    y_true = y_true.view(s0,-1)
    tp = torch.sum(y_pred * y_true, dim=1)
    fn = torch.sum((1-y_pred) * y_true, dim=1)
    fp = torch.sum(y_pred * (1-y_true), dim=1)
    
    adder = 1. if addone else 0.
    
    return (tp + adder) / (adder + tp + beta * fn + (1-beta) * fp)

class TverskyLoss(nn.Module):
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
    def forward(self, y_pred, y_true):
        y_pred = get_normalized_probs(y_pred)
        return torch.mean(1-tversky_index(y_pred, y_true, self.beta, addone=True))
    
    
"""
note: this Tversky Loss does not seem to be working properly
"""
class FocalTverskyLoss(nn.Module):
    def __init__(self, beta, gamma):
        super().__init__()
        self.beta = beta
        self.gamma = gamma
    def forward(self, y_pred, y_true):
        y_pred = get_normalized_probs(y_pred)
        return torch.mean((1. - tversky_index(y_pred, y_true, self.beta, addone=False)) ** self.gamma)