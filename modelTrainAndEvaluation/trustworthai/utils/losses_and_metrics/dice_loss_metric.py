from torchmetrics import Metric
import torch
from trustworthai.utils.losses_and_metrics import dice_loss

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
    

class SsnDiceMeanMetricWrapper(DiceLossMetric):
    def update(self, preds_dict, target: torch.Tensor):
        super().update(preds_dict['logit_mean'], target)

    def compute(self):
        return super().compute()