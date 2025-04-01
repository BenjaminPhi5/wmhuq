import torch

def get_fit_coords(mask, pad):
    mask = mask.squeeze()

    wheres = torch.where(mask !=0)
    zs = wheres[-3].min().item(), wheres[-3].max().item()
    xs = wheres[-2].min().item(), wheres[-2].max().item()
    ys = wheres[-1].min().item(), wheres[-1].max().item()
    
    shape = mask.shape
    
    zs = max(0, zs[0] - pad), min(shape[0], zs[1] + pad)
    xs = max(0, xs[0] - pad), min(shape[1], xs[1] + pad)
    ys = max(0, ys[0] - pad), min(shape[2], ys[1] + pad)

    return zs, xs, ys

def fit_to_mask(zs, xs, ys, img):
    if len(img.shape) == 4: # my presumption for the dataloader
        return img[:, zs[0]:zs[1]+1, xs[0]:xs[1]+1, ys[0]:ys[1]+1]
    elif len(img.shape == 5):
        return img[:, :, zs[0]:zs[1]+1, xs[0]:xs[1]+1, ys[0]:ys[1]+1]
    elif len(img.shape) == 3:
        return img[zs[0]:zs[1]+1, xs[0]:xs[1]+1, ys[0]:ys[1]+1]
    else:
        raise ValueError(f"img should be 3D, 4D or 5D, not {len(img.shape)}D")
