# TODO: complete

# see the on power jaccard losses paper, interesting
# https://www.scitepress.org/Papers/2021/103040/103040.pdf

# do standard jaccard loss as well.

# using power losses gives greater performance imporvement with smaller models.

from torch import nn
import torch


class PowerJaccardLoss(nn.Module):
    """
    p = 1 gives jaccard distance.
    p cannot be greater than 1 or less than 1
    """
    
    def __init__(self, power=1.5, epsilon=1e-8):
        # epsilon prevents 0 division.
        if power < 1 or power > 2:
            raise ValueError("1 <= p <= 2 must hold for convergence to desired value")
        
        self.p = power
        self.epsilon = epsilon
        
    def forward(self, y_pred, y_true):
        y_yhat = torch.sum(y_true * y_pred)
        
        numerator = y_yhat + self.epsilon
        
        denominator = torch.sum(torch.pow(y_true, self.p)) \
                    + torch.sum(torch.pow(y_pred, self.p)) \
                    - y_yhat \
                    + self.epsilon
        
        return 1. - (numerator / denominator)