from trustworthai.utils.losses_and_metrics.dice_loss import individual_dice
import torch

def log_cosh(l):
    return torch.log(torch.cosh(l))

def log_cosh_dice_loss(p_hat, y_true):
    combined = individual_dice(p_hat, y_true)
    combined = log_cosh(combined)
    return torch.mean(combined)