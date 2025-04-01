"""
fine grained control of transforms so that 
images and labels can be transformed in the same way where the images are
affinely transformed or deformed etc.
"""

"""
TODO: NOTE CURRENTLY THIS CONTAINS SOME DIRECTLY COPIED
CODE (IN THE RandomResizeCrop CLASS and another), NEED TO WRITE
MY OWN IMPLEMENTATION REALLY.

TODO NOTE for elastic deformations there are existing
pytorch libraries that I can try to use.

"""

from torchvision import transforms
import torchvision.transforms.functional as TF
from torchvision.transforms.functional import InterpolationMode
import numpy as np
from abc import ABC, abstractmethod
from collections.abc import Sequence
import math
import torch

"""
TODO: need to get my images proeprly inside the mask actually
and fix the noise so that it only applies to the brain ideally
as currently its all over the place and not great.

"""

# I need custom transforms so that I can transform both the image and the label in the same way
class PairedRandomTransform:
    def __init__(self, seed=None, rng=None, p=0.5):
        """
        seed: random seed to create a rng, or optionally, pass a preinstantiated numpy random generator (save
        instatiating a separate one for each transform)
        p: probability of applying the transform. must be between 0 and 1.
        """
        if p <= 0 or p > 1:
            raise ValueError("probability of application p must be in range 0 < p <= 1")
        self.p = p
        
        if seed == None:
            seed = 1307
        
        if rng==None:
            self.rng = np.random.default_rng(seed)
        else:
            self.rng = rng
        self.g = torch.Generator().manual_seed(seed)
    
    @abstractmethod
    def __call__(self, img, label):
        """
        takes an image and a label, and defines how the transform should behave for both
        (some transforms should transform the label, some should just pass it back)
        """
        pass


class RandomFlip(PairedRandomTransform):
    """
    randomly flip a brain scan either horizontally or vertically
    (pick the relevant flip depending on the orientation of the scan, so that
    the LR portions of the brain are flipped, but the brain is not flipped upside down
    
    by default my scans should be vertically flipped due to their orientation
    """
    def __init__(self, orientation="vertical", *args, **kwargs):
        """
        orientation: type of flip, vertical or horizontal only.
        """
        
        super().__init__(*args, **kwargs)
        if orientation != "vertical" and orientation != "horizontal":
            raise ValueError(f"Only horizontal or vertical accepted as orintation, not {orientation}")
            
        self.flip_func = TF.hflip if orientation == "horizontal" else TF.vflip
        
    def __call__(self, img, label):
        if self.rng.random() < self.p:
            img = self.flip_func(img)
            label = self.flip_func(label)
        
        return img, label
    
class GaussianBlur(PairedRandomTransform):
    """
    blurs image with random gaussian blur with a kernel
    of a given size and either a fixed standard deviation or
    if a tuple is given, kernel_sizes are randomly chosen from that
    range, if None then it is derived with a function of the kernel
    size, see the torch docs
    """
    def __init__(self, kernel_size, sigma, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.kernel_size = kernel_size
        self.sigma = sigma
        
        self.gen_sigma = False
        if isinstance(sigma, Sequence):
            self.gen_sigma = True
        
    def __call__(self, img, label):
        # NOTE: doesn't augment the label
        if self.rng.random() < self.p:
            if self.gen_sigma:
                sigmas = self.rng.random(2) * (self.sigma[1] - self.sigma[0]) + self.sigma[0]
                sigmas = (sigmas[0], sigmas[1])
                img = TF.gaussian_blur(img, kernel_size=self.kernel_size, sigma=sigmas)
            else:
                img = TF.gaussian_blur(img, self.kernel_size, self.sigma)
        return img, label
        
class GaussianNoise(PairedRandomTransform):
    def __init__(self, mean=0, sigma=0.25, *args, **kwargs):
        """
        gaussian noise with a universal standard deviation and mean
        as thats simplest to implement for now.
        """
        super().__init__(*args, **kwargs)
        
        self.mean = mean
        self.sigma = sigma
        
        # self.gen_sigma = False
        # if isinstance(sigma, Sequence):
        #     self.gen_sigma = True
        #     if len(sigma) != 2:
        #         raise ValueError("if sigma is a tuple, it should be a min max pair")
        #     if sigma[0] < 0 or sigma[1] < 0:
        #         raise ValueError("cannot have negative sigma")
        
    def __call__(self, img, label):
        if self.rng.random() < self.p:
#             if self.gen_sigma:
#                 # gen covariance matrix
#                 dims = img.size()
#                 sigmas = self.rng.random(dims) * (self.sigma[1] - self.sigma[0]) + self.sigma[0]
#                 covariance = torch.eye(dims) * sigmas # diagonal matrix
                
#                 img = torch.randn(img.size()) @ covariance + self.mean
            # else:
            noise = torch.randn(img.shape) * self.sigma + self.mean
            img2 = img + noise
            img2[:,1] = img[:,1]
                
        return img, label
        
class RandomResizeCrop(PairedRandomTransform):
    """
    crop an image and output it to the same size as the orignal image
    note images are expected to have a [...H,W] shape, so if the model is
    3D then the depth does indeed need to be at the front of the image
    this somwhat limits the kind of augmentations possible and could revisit this later
    """
    def __init__(self, scale=(0.08, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0), interpolation=InterpolationMode.BILINEAR, output_size=None, *args, **kwargs):
        # default params taken from standard toch object, see their documentation for details
        super().__init__(*args, **kwargs)
        self.scale = scale
        self.ratio = ratio
        self.interpolation = interpolation
        self.output_size=output_size
            
    def get_params(self, img):
        # based on code from the original Torch Object
        # https://pytorch.org/vision/stable/_modules/torchvision/transforms/transforms.html#RandomResizedCrop
        """Get parameters for ``crop`` for a random sized crop.

        Args:
            img (PIL Image or Tensor): Input image.
            scale (list): range of scale of the origin size cropped
            ratio (list): range of aspect ratio of the origin aspect ratio cropped

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
            sized crop.
        """
        width, height = TF.get_image_size(img)
        area = height * width
        scale = self.scale
        ratio = self.ratio

        log_ratio = torch.log(torch.tensor(ratio))
        # try 10 times to find a suitable crop
        for _ in range(10):
            target_area = area * torch.empty(1).uniform_(scale[0], scale[1]).item()
            aspect_ratio = torch.exp(torch.empty(1).uniform_(log_ratio[0], log_ratio[1])).item()

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                i = torch.randint(0, height - h + 1, size=(1,), generator=self.g).item()
                j = torch.randint(0, width - w + 1, size=(1,), generator=self.g).item()
                return i, j, h, w

        # Fallback to central crop
        in_ratio = float(width) / float(height)
        if in_ratio < min(ratio):
            w = width
            h = int(round(w / min(ratio)))
        elif in_ratio > max(ratio):
            h = height
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = width
            h = height
        i = (height - h) // 2
        j = (width - w) // 2
        return i, j, h, w
            
            
    def __call__(self, img, label):
        # NOTE: this is applied to both image and label!
        # also note, only does the crop in 2D.. not in the depth dimension
        # so might need to write my own version of this really
        # but that is a problem for later.
        if self.rng.random() < self.p:
            i, j, h, w = self.get_params(img)
            output_size = self.output_size
            if output_size == None:
                output_size = img.shape[-2:]
            img = TF.resized_crop(img, i, j, h, w, output_size, self.interpolation)
            label = TF.resized_crop(label, i, j, h, w, output_size, InterpolationMode.NEAREST)
            
        return img, label
        
class RandomAffine(PairedRandomTransform):
    """
    does rotation, shearing, translating as an all in out
    """
    def __init__(self, degrees=0, translate=None, scale=None, shear=None, 
                 interpolation=InterpolationMode.NEAREST,
                 fill=0, fillcolor=0, resample=None, center=None, *args, **kwargs):
        """
        NOTE in addition to the below, I expect degrees to be either value or float,
        shear and translate to ba a tuple or None, and scale to be an integer or None
        please see the docs for the random affine transformation for an explanation.
        the funcs in this file are mostly just wrappers on the functional versions of 
        the transforms implemented in pytorch. this message should go somewhere at the top
        of this file.....
        """
        super().__init__(*args, **kwargs)
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.interpolation = interpolation
        self.fill = fill
        self.center = center
        
        
    def __call__(self, img, label):
        if self.rng.random() < self.p:
            # generate the random transforms for each transform
            if self.degrees != 0:
                if isinstance(self.degrees, Sequence):
                    degrees = self.degrees
                else:
                    degrees = (-self.degrees, self.degrees)
                angle = self.rng.random() * (degrees[1] - degrees[0]) + degrees[0]
            else:
                angle = 0
                
                
            if self.shear == None:
                shear = 0
            else:
                params = self.rng.random(2) * (self.shear[1] - self.shear[0]) + self.shear[0]
                shear = (params[0], params[1])
                
            if self.translate == None:
                translate = (0,0)
            else:
                params = self.rng.random(2) * (self.translate[1] - self.translate[0]) + self.translate[0]
                translate = (params[0], params[1])
            
            if self.scale == None:
                scale = 1
            else:
                scale = self.rng.random() * (self.scale[1] - self.scale[0]) + self.scale[0]
                
            
            img = TF.affine(img, angle=angle, translate=translate, scale=scale, shear=shear, 
                            interpolation=self.interpolation, fill=self.fill, center=self.center)
            label = TF.affine(label, angle=angle, translate=translate, scale=scale, shear=shear, 
                            interpolation=InterpolationMode.NEAREST, fill=self.fill, center=self.center)
        
        return img, label
    
# FOR NOW TOO COMPLICATED, DUE TO NEEDING EITHER
# 1D OR 2D INPUT, SO SKIPPING...
# class RandomColourJitter(PairedRandomTransform):
#     """
#     TODO COMPLETE THIS AUGMENTATION IT SEEMS VERY USEFUL.
#     """
#     def __init__(self, brightness=None, contrast=None, saturation=None, hue=None, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.brightness = brightness
#         self.contrast = contrast
#         self.saturation = saturation
#         self.hue = hue
        
#     def get_params(self):
#         brightness = self.brightness
#         contrast = self.contrast
#         saturation = self.saturation
#         hue = self.hue
        
#         ## NOTE code for generating params is copied directly from the original 
#         ## torch ColorJitter transform object
#         ## should write own implementation as some point
#         """
#         Args:
#             brightness (tuple of float (min, max), optional): The range from which the brightness_factor is chosen
#                 uniformly. Pass None to turn off the transformation.
#             contrast (tuple of float (min, max), optional): The range from which the contrast_factor is chosen
#                 uniformly. Pass None to turn off the transformation.
#             saturation (tuple of float (min, max), optional): The range from which the saturation_factor is chosen
#                 uniformly. Pass None to turn off the transformation.
#             hue (tuple of float (min, max), optional): The range from which the hue_factor is chosen uniformly.
#                 Pass None to turn off the transformation.

#         Returns:
#             tuple: The parameters used to apply the randomized transform
#             along with their random order.
#         """
#         fn_idx = torch.randperm(4)

#         b = None if brightness is None else float(torch.empty(1).uniform_(brightness[0], brightness[1]))
#         c = None if contrast is None else float(torch.empty(1).uniform_(contrast[0], contrast[1]))
#         s = None if saturation is None else float(torch.empty(1).uniform_(saturation[0], saturation[1]))
#         h = None if hue is None else float(torch.empty(1).uniform_(hue[0], hue[1]))

#         return fn_idx, b, c, s, h
    
#     def __call__(self, img, label):
#         if self.rng.random() < self.p:
#             fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor = self.get_params()
            
#             # augmentation is applied only to image, not label
#             if len(img.shape) == 4:
#                 img = img.permute((1,0,2,3))
#             for fn_id in fn_idx:
#                 if fn_id == 0 and brightness_factor is not None:
#                     img = TF.adjust_brightness(img, brightness_factor)
#                 elif fn_id == 1 and contrast_factor is not None:
#                     img = TF.adjust_contrast(img, contrast_factor)
#                 elif fn_id == 2 and saturation_factor is not None:
#                     img = TF.adjust_saturation(img, saturation_factor)
#                 elif fn_id == 3 and hue_factor is not None:
#                     img = TF.adjust_hue(img, hue_factor)
            
#             if len(img.shape) == 4:
#                 img = img.permute((1,0,2,3))
        
#         return img, label


class CropZDim(PairedRandomTransform):
    def __init__(self, size, minimum, maximum, *args, **kwargs):
        """
            Arguments:
                size: number of z (depth) slices to include (will be taken contiguously)
                minimum:, minimum slice to start from
                maximum: maximum slice to finish at
                Note: assumes images are presented in (c, d, h, w) format
                
            Returns:
                a crop of the image along the depth axis and a label cropped identically.
        """
        
        super().__init__(p=1)
        self.size = size
        self.min_start = minimum
        self.max_start = maximum - size + 1 if maximum > 0 else -1
    
    def __call__(self, img, label):
        max_start = self.max_start
        if self.max_start < 0:
            max_start = img.shape[1] - self.size
        start_slice = self.rng.integers(self.min_start, max_start)
        end_slice = start_slice + self.size
        return img[:, start_slice:end_slice, :, :], label[:, start_slice:end_slice, :, :]
        
class NormalizeImg(PairedRandomTransform):
    def __init__(self, p=1, *args, **kwargs):
        """
        for p = 1 just always normalize the image
        """
        super().__init__(p=1, *args, **kwargs)
        
    def __call__(self, img, label):
        # add the specific p == 1 case as that is likely.
        if self.p == 1 or self.rng.random() < self.p:
            mean = torch.mean(img)
            std = torch.std(img)
            img = (img - mean) / std
        
        return img, label
    
    
class PairedCompose(transforms.Compose):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def __call__(self, img, label):
        for t in self.transforms:
            img, label = t(img, label)
        return img, label
    
    
class LabelSelect():
    def __init__(self, label_id):
        """
        converts pixels of a label to be
        1 when they match a particular label
        and 0 otherwise
        
        label_id the target label id 
        (e.g 1 for wmh in the wmh challenge dataset)
        """
        self.label_id = label_id
        
    def __call__(self, img, label):
        label = (label == self.label_id).type(torch.float32)
        
        return img, label
    
class DropMask():
    """
    assumes that the image has channels [mask, ...]
    removes the mask channel
    """
    def __call__(self, img, label):
        return img[1:], label
    
class PairedCentreCrop():
    def __init__(self, size):
        """
        crops the given image at the centre using the semantics
        of centre crop from the torch transforms centrecrop object.
        size can be (h,w) sequence, or int (h=w).
        """
        if isinstance(size, Sequence):
            self.output_size = size
        else:
            self.output_size = (size, size)
    
    def __call__(self, img, label):
        img = TF.center_crop(img, self.output_size)
        label = TF.center_crop(label, self.output_size)
        
        return img, label
