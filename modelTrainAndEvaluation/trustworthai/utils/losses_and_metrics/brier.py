import torch

def get_brier_loss(reduction="mean_sum"):
    # reduction = mean_sum | mean | sum | none
    if reduction == "mean_sum":
        mse_reduction = "none"
    else:
        mse_reduction = reduction
    def brier_loss(pred, target):
        pred = torch.nn.functional.softmax(pred, dim=1)[:,1]
        target = target.type(torch.float32)
        mse_loss = torch.nn.functional.mse_loss(pred, target, reduction=mse_reduction)
        if reduction == "mean_sum":
            bs = pred.shape[0]
            return mse_loss.view(bs, -1).sum(dim=1).mean(dim=0)
        else:
            return mse_loss
        
    return brier_loss