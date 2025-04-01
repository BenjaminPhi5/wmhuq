import torch
import torch.nn as nn
from trustworthai.utils.losses_and_metrics.per_individual_losses import DiceLossMetric

class SsnNetworkMeanLossWrapper(nn.Module):
    def __init__(self, loss_func):
        super().__init__()
        self.loss = loss_func
    def forward(self, result_dict, target):
        mean = result_dict['logit_mean']
        return self.loss(mean, target)
    
class SsnNetworkSampleLossWrapper(nn.Module):
    def __init__(self, loss_func):
        super().__init__()
        self.loss = loss_func
    def forward(self, result_dict, target):
        sample = result_dict['distribution'].rsample()
        target = target.to(sample.device)
        return self.loss(sample, target)
    
    
class SsnDiceMetricWrapper(DiceLossMetric):

    def update(self, preds_dict, target: torch.Tensor):
        super().update(preds_dict['mean'], target)

    def compute(self):
        return super().compute()