import torch
import torch.nn.functional as F

DIM1S = 224
DIM2S = 160

def crop_or_pad(array, w=DIM1S, d=DIM2S):
    # array of shape H, W, D and desired outcome shapes w and d.
    def crop_or_pad_dim(a, dim, desired_size):
        current_size = a.shape[dim]
        delta = desired_size - current_size
        if delta > 0:
            # Need to pad
            pad_left = delta // 2
            pad_right = delta - pad_left
            pad = [0] * (2 * a.dim())
            pad_index_left = 2 * (a.dim() - dim - 1)
            pad_index_right = pad_index_left + 1
            pad[pad_index_left] = pad_left
            pad[pad_index_right] = pad_right
            a = F.pad(a, pad)
        elif delta < 0:
            # Need to crop
            crop_left = (-delta) // 2
            crop_right = (-delta) - crop_left
            slices = [slice(None)] * a.dim()
            slices[dim] = slice(crop_left, current_size - crop_right)
            a = a[tuple(slices)]
        # else, no change needed
        return a

    # Adjust the W dimension
    array = crop_or_pad_dim(array, dim=1, desired_size=w)
    # Adjust the D dimension
    array = crop_or_pad_dim(array, dim=2, desired_size=d)
    return array
