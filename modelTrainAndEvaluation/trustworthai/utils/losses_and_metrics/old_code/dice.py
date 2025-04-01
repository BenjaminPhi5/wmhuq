# dice similarity coefficient
import torch
from torchmetrics import Metric

def dice(y_pred, y_true):
    denominator = torch.sum(y_pred) + torch.sum(y_true)
    numerator = 2. * torch.sum(torch.logical_and(y_pred, y_true))
    return numerator / denominator


class DiceMetric(Metric):
    def update(self, *args, **kwargs):
        pass
    
    def compute(self, y_pred, y_true):
        return dice(y_pred, y_true)
    