import numpy as np
from twaidata.MRI_preprep.io import load_nii_img, save_nii_img

def get_brain_mean_std(whole_img3D, cutoff=0.01):
    """
        get mean and starndard deviation of the brain pixels, 
        where brain pixels are all those pixels that are > cutoff 
        in intensity value.
        returns the mean, the std and the locations where the brain is present.
    """
    brain_locs = whole_img3D > cutoff # binary map, 1 for included
    brain3D = whole_img3D[brain_locs]
    
    mean = np.mean(brain3D)
    std = np.std(brain3D)
    
    return mean, std, brain_locs

def normalize_brain(whole_img3D, cutoff=0.01):
    """
    whole_img3D: numpy array of a brain scan
    
    normalize brain pixels using global mean and std.
    only pixels > cutoff in intensity are included.
    """
    mean, std, brain_locs = get_brain_mean_std(whole_img3D, cutoff)
    whole_img3D[brain_locs] = (whole_img3D[brain_locs] - mean) / std
    
    
def normalize(img_file, out_file):
    img, header = load_nii_img(img_file)
    img = img.squeeze()
    normalize_brain(img) # in place operation

    save_nii_img(out_file, img, header)
