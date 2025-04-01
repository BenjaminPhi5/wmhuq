"""
Wrappers ontop of MONAI that makes things easy for me
"""
from monai.transforms import MapTransform
from abc import abstractmethod
from torchvision.transforms.functional import resize
from torchvision.transforms import InterpolationMode
import numpy as np
from monai.transforms import RandRotated
from monai.transforms import RandAffined
from monai.transforms import RandGaussianNoised
from monai.transforms import RandGaussianSmooth
from monai.transforms import RandScaleIntensityd
from monai.transforms import RandAdjustContrastd
from monai.transforms import RandFlipd
from monai.transforms import Compose
from twaibrain.braintorch.augmentation.label_format import OneVRest, OneHotEncoder
from torch import float32, long, cat
import torch.nn.functional as F

class MonaiAugmentationExtender(MapTransform):
    def __init__(self, p:float, dims:int=2, keys=['image', 'mask', 'label'], mode=None, *args, **kwargs):
        super().__init__(keys=keys, *args, **kwargs)
        assert dims == 2 or dims == 3
        assert 0 <= p <= 1
        self.dims = dims # 2 for 2D, 3 for 3D
        self.mode= mode # modes are used where we treat labels and images differently.
        self.p = p # p is probability of applying the transform.

    @abstractmethod
    def __call__(self, data):
        pass

class MonaiAugmentationWrapper(MonaiAugmentationExtender):
    # needs to have base_augmentation defined in the constructor
    def __call__(self, data):
        return self.base_augmentation(data)


class RotationAugment(MonaiAugmentationWrapper):
    def __init__(self, axial_only=False, *args, **kwargs):
        super().__init__(*args, mode=['bilinear', 'nearest', 'nearest'], **kwargs)

        pi = np.pi
        
        degrees_3D=(-30 * pi/180,30 * pi/180)
        degrees_2D =(-pi,pi)

        if axial_only and self.dims == 3:
            range_x = range_y = (0,0)
            range_z = degrees_2D

        elif self.dims == 3:
            range_x = range_y = range_z = degrees_3D

        elif self.dims == 2:
            range_x = range_y = range_z = degrees_2D

        self.base_augmentation = RandRotated(
            keys=self.keys,
            mode=self.mode,
            range_x=range_x,
            range_y=range_y,
            range_z=range_z,
            keep_size=True,
            prob=self.p
        )


class AffineAugment(MonaiAugmentationExtender):
    def __init__(self, spatial_size, axial_only=False, *args, **kwargs):
        """
        spatial size should be the output size I want I think? (or just the size of the data
        , i.e dont change the size
        I should do the affine translation and then centre crop perhaps? I'm not really sure how the
        spatial dim stuff works at the moment, and when I try 3D I should think about this.
        """
        super().__init__(*args, mode=['bilinear', 'nearest', 'nearest'], **kwargs)
        self.spatial_size = spatial_size

        pi = np.pi

        ### setup rotation range
        # degrees_3D=(-30 * pi/180,30 * pi/180) # for isotropic
        degrees_2D = (-pi,pi)
        degrees_3D = degrees_2D # my data is anisotropic 3D for now.

        if axial_only and self.dims == 3:
            range_x = range_z = (0,0)
            range_y = degrees_2D

        elif self.dims == 3:
            range_x = range_y = range_z = degrees_3D

        elif self.dims == 2:
            range_x = range_y = range_z = degrees_2D

        self.rotate_range = (*range_x, *range_y, *range_z)

        ### setup translation range
        translation_scale = 0.1
        self.translate_range = [(-s * translation_scale, s * translation_scale) for s in spatial_size]

        ### setup scale range
        scale_min = -0.3
        scale_max = 0.4
        self.scale_range = (scale_min, scale_max, scale_min, scale_max, scale_min, scale_max)

        ### setup shear range
        shear_angle = 18 * pi/180
        if axial_only and self.dims == 3:
            self.shear_range = (0, 0, -shear_angle, shear_angle, 0,0)
        else:
            self.shear_range = (-shear_angle, shear_angle, -shear_angle, shear_angle, -shear_angle)
        

    def __call__(self, data):
        # decide whether to call augmentation:
        do_rotation = np.random.uniform(0,1) < self.p 
        do_scale = np.random.uniform(0,1) < self.p
        do_translate = np.random.uniform(0,1) < self.p
        do_shear = np.random.uniform(0,1) < self.p

        if not do_rotation and not do_scale and not do_translate and not do_shear:
            return data

        rotate_range = self.rotate_range if do_rotation else None
        # print(rotate_range)
        scale_range = self.scale_range if do_scale else None
        translate_range = self.translate_range if do_translate else None
        shear_range = self.shear_range if do_shear else None
        
        augment = RandAffined(
            keys=self.keys,
            mode=self.mode,
            spatial_size=self.spatial_size, 
            prob=1,
            rotate_range=rotate_range,
            scale_range=scale_range,
            translate_range=translate_range,
            shear_range=shear_range,
            padding_mode='zeros'
        )
        
        return augment(data)

class GaussianNoiseAugment(MonaiAugmentationWrapper):
    def __init__(self, keys=['image'], *args, **kwargs):
        super().__init__(*args, keys=keys, **kwargs)
        self.base_augmentation = RandGaussianNoised(
            keys=self.keys,
            prob=self.p,
            mean=0,
            std=0.1 # this parameter is the top end of the rane that the noise is sampled from (I think)
        )


class GaussianBlurAugment(MonaiAugmentationExtender):
    def __init__(self, modality_p, keys=['image'], *args, **kwargs):
        super().__init__(*args, keys=keys, **kwargs)
        self.modality_p = modality_p
        self.base_augmentation = RandGaussianSmooth(
            prob=self.modality_p,
            sigma_x=(0.5,1.5),
            sigma_y=(0.5,1.5),
            sigma_z=(0.5,1.5),
        )
        # print(self.p)
        # print(self.modality_p)

    def __call__(self, data):
        if np.random.uniform(0,1) > self.p:
            # print("returning")
            return data

        for key in self.keys:
            key_data = data[key].clone()
            for channel in range(key_data.shape[0]):
                channel_data = key_data[channel].unsqueeze(0)
                channel_data = self.base_augmentation(channel_data)
                key_data[channel] = channel_data

            data[key] = key_data
        return data

class BrightnessAugment(MonaiAugmentationWrapper):
    def __init__(self, keys=['image'], *args, **kwargs):
        super().__init__(*args, keys=keys, **kwargs)
        # monai uses v = v * (1 + factor)
        # I want to multiply by range 0.7, to 1.3,, so in range -0.3, to 0.3
        self.base_augmentation = RandScaleIntensityd(
            keys=self.keys,
            factors=(-0.3,0.3),
            prob=self.p,
        )

class ContrastAugment(MonaiAugmentationWrapper):
    def __init__(self, keys=['image'], *args, **kwargs):
        super().__init__(*args, keys=keys, **kwargs)
        self.base_augmentation = RandAdjustContrastd(
            keys=self.keys,
            prob=self.p,
            gamma=(0.65,1.5)
        )


class LowResolutionSimulationAugmentation(MonaiAugmentationExtender):
    def __init__(self, modality_p, keys=['image'], *args, **kwargs):
        super().__init__(*args, keys=keys, **kwargs)
        self.modality_p = modality_p

    def __call__(self, data):
        # decide whether to call augmentation:
        if np.random.uniform(0,1) > self.p:
            return data

        # generate downsample factor
        factor = np.random.uniform(0.5, 1)
        # print(factor)

        for key in self.keys:
            if key not in data:
                raise KeyError(f"key {key} not found and allow_missing_keys==False for this augmentation")

            key_data = data[key].clone()
            for channel in range(key_data.shape[0]):
                if np.random.uniform(0,1) < self.modality_p:
                    # print(key_data[channel].shape)
                    channel_data = key_data[channel].unsqueeze(0)
                    spatial_dims = np.array(channel_data.shape[-2:]) # only take the last two dims, so don't downsample the already low resolution z plane.
                    downsampled_size = np.round(spatial_dims * factor).astype(np.int32)
        
                    downsampled = resize(channel_data, downsampled_size, interpolation=InterpolationMode.NEAREST)
                    upsampled = resize(downsampled, spatial_dims, interpolation=InterpolationMode.BICUBIC, antialias=True)
                    # print(upsampled.squeeze().shape)
                    key_data[channel] = upsampled.squeeze()
            data[key] = key_data
            
        return data


class GammaAugmentation(MonaiAugmentationExtender):
    def __init__(self, *args, keys=['image'], allow_invert=True, **kwargs):
        super().__init__(*args, keys=keys, **kwargs)
        self.allow_invert=allow_invert
    
    def __call__(self, data):
        # decide whether to call augmentation:
        if np.random.uniform(0,1) > self.p:
            return data

        gamma = np.random.uniform(0.7, 1.5)

        invert = self.allow_invert and np.random.uniform(0,1) < 0.15

        for key in self.keys:
            if key not in data:
                raise KeyError(f"key {key} not found and allow_missing_keys==False for this augmentation")

            key_data = data[key]

            # 0, 1 scale
            drange = (key_data.min(), key_data.max())
            key_data = (key_data - drange[0]) / (drange[1] - drange[0])

            # gamma scale
            if invert:
                key_data = 1 - (1-key_data).pow(gamma)
            else:
                key_data = key_data.pow(gamma)

            # original range scale
            key_data = key_data * (drange[1] -drange[0]) + drange[0]
            
            data[key] = key_data

        return data

class MirrorAugment(MonaiAugmentationExtender):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.flip0 = RandFlipd(keys=self.keys, prob=self.p, spatial_axis=0)
        self.flip1 = RandFlipd(keys=self.keys, prob=self.p, spatial_axis=1)
        self.flip2 = RandFlipd(keys=self.keys, prob=self.p, spatial_axis=2)

    def __call__(self, data):
        # todo: is this nessesary? what happens if I ignore this?
        if self.dims == 2:
            return self.flip0(data)
        else:
            return self.flip0(self.flip1(self.flip2(data)))


class SetDtype():
    def __init__(self, keys, dtypes):
        self.keys = keys
        self.dtypes = dtypes

    def __call__(self, data):
        for i, key in enumerate(self.keys):
            data[key] = data[key].type(self.dtypes[i])

        return data
        

class SetDtypeImageLabelPair():
    def __init__(self, keys, dtypes):
        self.keys = keys
        self.dtypes = dtypes

    def __call__(self, data):
        for i, key in enumerate(self.keys):
            data[key] = data[key].type(self.dtypes[i])

        return cat([data['image'], data['mask']]), data['label']


class MonaiPairedPadToShape2d:
    def __init__(self, target_shape, padding_mode="constant", padding_value=0):
        self.target_shape = target_shape
        self.padding_mode = padding_mode
        self.padding_value = padding_value

    def __call__(self, data):
        img = cat([data['image'], data['mask']])
        label = data['label']
        
        _, h, w = img.shape
        target_h, target_w = self.target_shape

        pad_h = max(0, target_h - h)
        pad_w = max(0, target_w - w)

        padding = (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2)
        img = F.pad(img, padding, mode=self.padding_mode, value=self.padding_value)
        label = F.pad(label, padding, mode=self.padding_mode, value=self.padding_value)

        data = {
            'image':img[0:2],
            'mask':img[2].unsqueeze(0),
            'label':label
        }
        
        return data

class MonaiPairedPadToShape3d:
    def __init__(self, target_shape, padding_mode="constant", padding_value=0):
        self.target_shape = target_shape
        self.padding_mode = padding_mode
        self.padding_value = padding_value

    def __call__(self, data):
        img = cat([data['image'], data['mask']])
        label = data['label']
        
        _, d, h, w = img.shape
        target_d, target_h, target_w = self.target_shape

        pad_d = max(0, target_d - d)
        pad_h = max(0, target_h - h)
        pad_w = max(0, target_w - w)

        padding = (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2, pad_d // 2, pad_d - pad_d // 2)
        img = F.pad(img, padding, mode=self.padding_mode, value=self.padding_value)
        label = F.pad(label, padding, mode=self.padding_mode, value=self.padding_value)

        data = {
            'image':img[0:2],
            'mask':img[2].unsqueeze(0),
            'label':label
        }
        
        return data

class MonaiCropAndPadToShape3d:
    def __init__(self, target_shape, padding_mode="constant", padding_value=0):
        self.target_shape = target_shape
        self.padding_mode = padding_mode
        self.padding_value = padding_value

    def __call__(self, data):
        img = data['image']
        mask = data['mask']
        label = data['label']
        
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
        mask = mask[:,cropping[0]:d-cropping[1],cropping[2]:h-cropping[3], cropping[4]:w-cropping[5]]
        label = label[:,cropping[0]:d-cropping[1],cropping[2]:h-cropping[3], cropping[4]:w-cropping[5]]
        
        img = F.pad(img, padding, mode=self.padding_mode, value=self.padding_value)
        mask = F.pad(mask, padding, mode=self.padding_mode, value=self.padding_value)
        label = F.pad(label, padding, mode=self.padding_mode, value=self.padding_value)

        data = {
            'image':img,
            'mask':mask,
            'label':label
        }
        
        return data
        

def get_nnunet_transforms(axial_only, dims, out_spatial_dims=(192, 224), allow_invert=True, one_hot_encode=True, num_classes=None, target_class=None):
    """
    axial only: only apply rotation and shearing in the axial plane, for 3D data
    out_spatial_dims: the output image size from the affine transformation
    dims: 2 for 2D, 3 for 3D
    allow_invert: allow gamma augmentation to be computed on inverse intensities

    all clases and the below p values uses parameters hardcoded from the nn-Unet paper.

    if one_hot_encode, applies one_hot_encoding to the labels prior to augmentation, where num_classes specifies the number of classes
    if not, uses target class to apply one v rest
    """
    assert dims==2 or dims==3

    assert one_hot_encode or target_class != None
    if one_hot_encode:
        assert num_classes != None

    if dims == 2:
        resizer = MonaiPairedPadToShape2d(out_spatial_dims)
    else:
        resizer = MonaiCropAndPadToShape3d(out_spatial_dims)

    label_formatter = OneHotEncoder(num_classes, 'label') if one_hot_encode else OneVRest(target_class, 'label')

    return Compose([
        resizer,
        AffineAugment(p=0.2, spatial_size=out_spatial_dims, dims=dims, axial_only=axial_only),
        GaussianNoiseAugment(p=0.15, dims=dims),
        GaussianBlurAugment(p=0.2, modality_p=0.5, dims=dims),
        BrightnessAugment(p=0.15, dims=dims),
        ContrastAugment(p=0.15, dims=dims),
        LowResolutionSimulationAugmentation(p=0.25, modality_p=0.5, dims=dims),
        GammaAugmentation(p=0.15, allow_invert=allow_invert, dims=dims),
        MirrorAugment(p=0.5, dims=dims),
        # label_formatter, # put label formatter at the end otherwise we get affine filling background channel with 0's when they should be ones
        SetDtypeImageLabelPair(keys=['image', 'mask', 'label'], dtypes=[float32, long, long])
    ])

def get_simple_transforms(axial_only, dims, out_spatial_dims=(192, 224), allow_invert=True, one_hot_encode=True, num_classes=None, target_class=None):
    """
    axial only: only apply rotation and shearing in the axial plane, for 3D data
    out_spatial_dims: the output image size from the affine transformation
    dims: 2 for 2D, 3 for 3D
    allow_invert: allow gamma augmentation to be computed on inverse intensities

    all clases and the below p values uses parameters hardcoded from the nn-Unet paper.

    if one_hot_encode, applies one_hot_encoding to the labels prior to augmentation, where num_classes specifies the number of classes
    if not, uses target class to apply one v rest
    """
    assert dims==2 or dims==3

    assert one_hot_encode or target_class != None
    if one_hot_encode:
        assert num_classes != None

    if dims == 2:
        resizer = MonaiPairedPadToShape2d(out_spatial_dims)
    else:
        resizer = MonaiCropAndPadToShape3d(out_spatial_dims)

    label_formatter = OneHotEncoder(num_classes, 'label') if one_hot_encode else OneVRest(target_class, 'label')

    return Compose([
        resizer,
        AffineAugment(p=0.4, spatial_size=out_spatial_dims, dims=dims, axial_only=axial_only),
        MirrorAugment(p=0.5, dims=dims),
        # label_formatter, # put label formatter at the end otherwise we get affine filling background channel with 0's when they should be ones
        SetDtypeImageLabelPair(keys=['image', 'mask', 'label'], dtypes=[float32, long, long])
    ])


def get_val_transforms( dims, out_spatial_dims=(192, 224), allow_invert=True, one_hot_encode=True, num_classes=None, target_class=None):
    """
    axial only: only apply rotation and shearing in the axial plane, for 3D data
    out_spatial_dims: the output image size from the affine transformation
    dims: 2 for 2D, 3 for 3D
    allow_invert: allow gamma augmentation to be computed on inverse intensities

    all clases and the below p values uses parameters hardcoded from the nn-Unet paper.

    if one_hot_encode, applies one_hot_encoding to the labels prior to augmentation, where num_classes specifies the number of classes
    if not, uses target class to apply one v rest
    """
    assert dims==2 or dims==3

    assert one_hot_encode or target_class != None
    if one_hot_encode:
        assert num_classes != None

    if dims == 2:
        resizer = MonaiPairedPadToShape2d(out_spatial_dims)
    else:
        resizer = MonaiCropAndPadToShape3d(out_spatial_dims)

    label_formatter = OneHotEncoder(num_classes, 'label') if one_hot_encode else OneVRest(target_class, 'label')

    return Compose([
        resizer,
        # MirrorAugment(p=0.5, dims=dims),
        # label_formatter, # put label formatter at the end otherwise we get affine filling background channel with 0's when they should be ones
        SetDtypeImageLabelPair(keys=['image', 'label', 'mask'], dtypes=[float32, long, long])
    ])
