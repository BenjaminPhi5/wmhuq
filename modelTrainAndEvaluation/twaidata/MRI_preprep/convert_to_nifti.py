import nibabel as nib
import numpy as np

def convert_to_nifti_format(hdr_in_path, out_path):
    """
    converts from analyze format to nifti (I think its analyze... regardless, I need it for ADNI)
    """
    nib_img = nib.load(hdr_in_path)
    
    # the .img file isn't actually loaded
    # until we request the data, which causes a problem when trying to save the data...
    # but here I convert all the images to float 32 anyway and create a new image which avoids
    # the problem
    data = nib_img.get_fdata().astype(np.float32)
    new_nib_img = nib.Nifti1Image(data, affine=nib_img.affine)
    
    #
    nib.save(new_nib_img, out_path)
