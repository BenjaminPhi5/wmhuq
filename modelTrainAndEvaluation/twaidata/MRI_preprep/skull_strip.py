from pyrobex.robex import robex
import nibabel as nib
import numpy as np

def skull_strip(t1_image):
    """
    image: filepath to T1, OR nibabel image object
    """
    if type(t1_image) == str:
        t1_image = nib.load(t1_image)

    stripped, mask = robex(t1_image)
    # mask = nib.nifti1.Nifti1Image(mask.get_fdata(), affine=t1_image.affine, header=t1_image.header)

    return stripped, mask

def skull_strip_and_save(t1_path, out_path, mask_path):
    image = nib.load(t1_path)

    stripped, mask = skull_strip(image)

    nib.save(stripped, out_path)
    nib.save(mask, mask_path)

def apply_mask(image, mask):
    # assmues image and mask are nibabel images.
    image_data = image.get_fdata()
    mask_data = mask.get_fdata()
    new_image = nib.nifti1.Nifti1Image(image_data * mask_data, affine=image.affine, header=image.header)
    return new_image

def apply_mask_and_save(image_path, mask_path, out_path):
    image = nib.load(image_path)
    mask = nib.load(mask_path)
    
    masked_image = apply_mask(image, mask)
    
    nib.save(masked_image, out_path)

def create_mask_from_background_value(image_path, mask_save_path, background=0.0):
    image = nib.load(image_path)
    image_data = image.get_fdata()
    mask = image_data != background
    mask = mask.astype(np.float32)
    mask_image = nib.nifti1.Nifti1Image(mask, affine=image.affine, header=image.header)
    nib.save(mask_image, mask_save_path)