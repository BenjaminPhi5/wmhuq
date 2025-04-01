import torch.nn as nn
import torch
from torchmetrics import Metric

def two_class_prob(p_hat):
    p_hat = torch.nn.functional.softmax(p_hat, dim=1)
    p_hat = p_hat[:,1,:] # select class 0
    return p_hat

def log_cosh(l):
    return torch.log(torch.cosh(l))

def individual_dice(p_hat, y_true):
    p_hat = two_class_prob(p_hat)
    s0 = p_hat.shape[0]
    p_hat = p_hat.view(s0,-1)
    y_true = y_true.view(s0,-1)
    numerator = torch.sum(2. * p_hat * y_true, dim=1) + 1.
    denominator = torch.sum(y_true + p_hat, dim=1) + 1.
    combined = 1. - (numerator/denominator)
    return combined
    
def dice_loss(p_hat, y_true):
    combined = individual_dice(p_hat, y_true)
    return torch.mean(combined)

def log_cosh_dice_loss(p_hat, y_true):
    combined = individual_dice(p_hat, y_true)
    combined = log_cosh(combined)
    return torch.mean(combined)


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
        y_pred = two_class_prob(y_pred)
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
        y_pred = two_class_prob(y_pred)
        return torch.mean((1. - tversky_index(y_pred, y_true, self.beta, addone=False)) ** self.gamma)
    
    

"""
metric for tracking dice loss when using other losses to train with.
"""
class DiceLossMetric(Metric):
    is_differentiable = False
    higher_is_better = True
    full_state_update = False
    def __init__(self):
        super().__init__()
        self.add_state("correct", default=torch.tensor(0, dtype=torch.float32), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0, dtype=torch.float32), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):

        #update = torch.sum(preds==1)
        self.correct += dice_loss(preds, target)
        self.total += 1 #target.numel() # total is batches now, so is only correct if batches are the same size.
        # print("GOT HERE: ", update)

    def compute(self):
        return self.correct.float() / self.total