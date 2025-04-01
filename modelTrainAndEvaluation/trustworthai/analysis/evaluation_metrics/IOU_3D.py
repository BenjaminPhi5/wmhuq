import torch

def IOU(bs, cs, bs_are_targets=False, cs_are_targets=True):
    # for computing on a 3D image
    if not bs_are_targets:
        bs = bs.argmax(dim=1)
    if not cs_are_targets:
        cs = cs.argmax(dim=1)
    intersection = torch.sum(bs * cs)
    union = torch.logical_or(bs, cs).sum()
    return intersection / union

def all_samples_iou(bs, cs):
    ious = []
    for i in range(len(bs)):
        ious.append(IOU(bs, cs, True, True))
    return torch.Tensor(ious).mean()