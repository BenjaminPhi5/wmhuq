from torch import float32, long, cat
import torch.nn.functional as F
from twaibrain.braintorch.augmentation.nnunet_augmentations import *

class MonaiPairedPadToShape2d_V2:
    def __init__(self, target_shape, padding_mode="constant", padding_value=0, keys=['image', 'mask', 'label']):
        self.target_shape = target_shape
        self.padding_mode = padding_mode
        self.padding_value = padding_value
        self.keys = keys

    def __call__(self, data):
        for key in self.keys():
            img = data[key]
        
            _, h, w = img.shape
            target_h, target_w = self.target_shape
    
            pad_h = max(0, target_h - h)
            pad_w = max(0, target_w - w)
    
            padding = (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2)
            img = F.pad(img, padding, mode=self.padding_mode, value=self.padding_value)
    
            data[key] = img
        
        return data

class MonaiCropAndPadToShape3d_V2:
    def __init__(self, target_shape, padding_mode="constant", padding_value=0, keys=['image', 'mask', 'label']):
        self.target_shape = target_shape
        self.padding_mode = padding_mode
        self.padding_value = padding_value
        self.keys = keys

    def __call__(self, data):
        for key in self.keys:
            img = data[key]
            
            _, d, h, w = img.shape
            target_d, target_h, target_w = self.target_shape
    
            crop_d = max(0, d - target_d)
            crop_h = max(0, h - target_h)
            crop_w = max(0, w - target_w)
            
            pad_d = max(0, target_d - d)
            pad_h = max(0, target_h - h)
            pad_w = max(0, target_w - w)
    
            cropping = (crop_d // 2, crop_d - crop_d // 2, crop_h // 2, crop_h - crop_h // 2, crop_w, crop_w - crop_w // 2)
            padding = (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2, pad_d // 2, pad_d - pad_d // 2)
            
            img = img[:,cropping[0]:d-cropping[1],cropping[2]:h-cropping[3], cropping[4]:w-cropping[5]]
            
            img = F.pad(img, padding, mode=self.padding_mode, value=self.padding_value)
    
            data[key] = img
        
        return data

class SetImageDtype:
    def __init__(self, keys, dtypes):
        self.keys = keys
        self.dtypes = dtypes

    def __call__(self, data):
        for i, key in enumerate(self.keys):
            data[key] = data[key].type(self.dtypes[i])

        return data

def get_default_nnunet_transforms(axial_only, dims, out_spatial_dims=(80, 192, 160), allow_invert=True, image_keys=['image'], label_keys=['label', 'mask']):
    """
    axial only: only apply rotation and shearing in the axial plane, for 3D data
    out_spatial_dims: the output image size from the affine transformation
    dims: 2 for 2D, 3 for 3D
    allow_invert: allow gamma augmentation to be computed on inverse intensities

    all classes and the below p values uses parameters hardcoded from the nn-Unet paper.
    """
    assert dims==2 or dims==3
    combined_keys = image_keys + label_keys
    
    if dims == 2:
        resizer = MonaiPairedPadToShape2d_V2(out_spatial_dims, keys=combined_keys)
    else:
        resizer = MonaiCropAndPadToShape3d_V2(out_spatial_dims, keys=combined_keys)

    return Compose([
        resizer,
        AffineAugment(p=0.2, spatial_size=out_spatial_dims, dims=dims, axial_only=axial_only, keys=combined_keys),
        GaussianNoiseAugment(p=0.15, dims=dims, keys=image_keys),
        GaussianBlurAugment(p=0.2, modality_p=0.5, dims=dims, keys=image_keys),
        BrightnessAugment(p=0.15, dims=dims, keys=image_keys),
        ContrastAugment(p=0.15, dims=dims, keys=image_keys),
        LowResolutionSimulationAugmentation(p=0.25, modality_p=0.5, dims=dims, keys=image_keys),
        GammaAugmentation(p=0.15, allow_invert=allow_invert, dims=dims, keys=image_keys),
        MirrorAugment(p=0.5, dims=dims, keys=combined_keys),
        SetImageDtype(keys=combined_keys, dtypes=[float32 for _ in range(len(image_keys))] + [long for _ in range(len(label_keys))])
    ])

def get_val_transforms(dims, out_spatial_dims=(80, 192, 160), image_keys=['image'], label_keys=['label', 'mask']):
    """
    just resize and set output dtype.
    """
    assert dims==2 or dims==3

    combined_keys = image_keys + label_keys
    
    if dims == 2:
        resizer = MonaiPairedPadToShape2d_V2(out_spatial_dims, keys=combined_keys)
    else:
        resizer = MonaiCropAndPadToShape3d_V2(out_spatial_dims, keys=combined_keys)

    return Compose([
        resizer,
        SetImageDtype(keys=combined_keys, dtypes=[float32 for _ in range(len(image_keys))] + [long for _ in range(len(label_keys))])
    ])