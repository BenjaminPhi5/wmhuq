import torch
import torch.nn.functional as F

def torch_resample(img, out_spacing, orig_spacing, is_label=False):
    img = img.unsqueeze(0)
    
    out_spacing = torch.Tensor(out_spacing).flip(0)
    orig_spacing = torch.Tensor(orig_spacing).flip(0)
    orig_size = torch.Tensor([*img.shape[-3:]])
    
    out_size = orig_size * orig_spacing / out_spacing
    out_size = out_size.type(torch.int32)

    img = F.interpolate(img, size=out_size.tolist(), mode='trilinear' if not is_label else 'nearest')
    img = img.squeeze(0)

    return img
