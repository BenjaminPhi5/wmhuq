import nibabel as nib
import numpy as np

FORMAT = '.nii.gz'

def load_nii_img(filename):
    """
    load a '.nii.gz' file and return
    a numpy array, along with the image header
    """
    
    if FORMAT not in filename:
        filename += FORMAT
    img_obj = nib.load(filename)
    img_data = img_obj.get_fdata()
    header = img_obj.header
    return img_data, header


def save_nii_img(filename, data, header):
    """
    save a numpy array as a '.nii.gz' file
    filename: name of file to be saved
    data: the numpy array of the file
    header: the header from the original image
    """
    if FORMAT not in filename:
        filename += FORMAT
    
    new_image = nib.nifti1.Nifti1Image(data, None, header=header)
    
    nib.save(new_image, filename)
    