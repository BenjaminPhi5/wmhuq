# see paper:
# Tversky loss function for image segmentation using 3D fully convolutional deep networks
# https://arxiv.org/pdf/1706.05721.pdf
import torch
import torch.nn as nn

class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.7, smooth=1e-8):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
    
    def forward(self, y_pred, y_true):
        y_pred = torch.flatten(y_pred)
        y_true = torch.flatten(y_true)
        numerator = torch.sum(y_pred * y_true)                       # true positives
        denominator = (
                    numerator 
                    + self.alpha * torch.sum(y_pred * (1.-y_true)) # false positives 
                    + self.beta  * torch.sum((1.-y_pred) * y_true) # false negatives
                    + self.smooth
        )
        # NOTE: THIS ASSUMES THAT THE Y_PRED IS POSITIVE (and a probability)
        tversky = (numerator / denominator)
        return 1. - tversky