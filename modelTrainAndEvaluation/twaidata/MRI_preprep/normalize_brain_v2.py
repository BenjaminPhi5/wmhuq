import numpy as np
import nibabel as nib

def normalize_brain(image, mask, lower_percentile=0, upper_percentile=1):
    image = image.copy()
    brain_locs = image[mask]
    
    if lower_percentile > 0 or upper_percentile < 1:
        print("normalizing with percentiles: ", lower_percentile, upper_percentile)
        brain_locs = brain_locs.flatten()
        sorted_indices = np.argsort(brain_locs)
        num_brain_voxels = len(sorted_indices)
        #print(num_brain_voxels)

        lower_index = int(lower_percentile*num_brain_voxels)
        upper_index = int(upper_percentile*num_brain_voxels)

        retained_indices = sorted_indices[lower_index:upper_index]
        #print(len(retained_indices)/num_brain_voxels)
        
        brain_locs = brain_locs[lower_index:upper_index]
    else:
        print("no percentiles used for normalization")

    mean = np.mean(brain_locs)
    std = np.std(brain_locs)
    
    print(mean, std)

    image[mask] = (image[mask] - mean) / std

    return image

def nib_normalize_brain(nib_image, nib_mask, lower_percentile=0, upper_percentile=1):
    image_data = nib_image.get_fdata()
    mask_data = nib_mask.get_fdata().astype(bool)

    image_data = normalize_brain(image_data, mask_data, lower_percentile, upper_percentile)

    return nib.nifti1.Nifti1Image(image_data, affine=nib_image.affine, header=nib_image.header)

def normalize(img_path, mask_path, out_path, lower_percentile=0, upper_percentile=1.0, background=0.0):
    nib_image = nib.load(img_path)
    
    if mask_path is None:
        normed_image = nib_normalize_brain_without_mask(nib_image, background)
    else:
        nib_mask = nib.load(mask_path)
    
        normed_image = nib_normalize_brain(nib_image, nib_mask, lower_percentile, upper_percentile)
    
    nib.save(normed_image, out_path)
    
    
def get_brain_mean_std_without_mask(whole_img3D, background=0.0):
    """
        get mean and starndard deviation of the brain pixels, 
        where brain pixels are all those pixels that are > cutoff 
        in intensity value.
        returns the mean, the std and the locations where the brain is present.
    """
    brain_locs = whole_img3D != background # binary map, 1 for included
    brain3D = whole_img3D[brain_locs]
    
    mean = np.mean(brain3D)
    std = np.std(brain3D)
    
    return mean, std, brain_locs

def normalize_brain_without_mask(whole_img3D, background=0.0):
    """
    whole_img3D: numpy array of a brain scan
    
    normalize brain pixels using global mean and std.
    only pixels > cutoff in intensity are included.
    """
    mean, std, brain_locs = get_brain_mean_std_without_mask(whole_img3D, background)
    whole_img3D[brain_locs] = (whole_img3D[brain_locs] - mean) / std

    return whole_img3D

def nib_normalize_brain_without_mask(nib_image, background=0.0):
    image_data = nib_image.get_fdata()

    image_data = normalize_brain_without_mask(image_data, background)

    return nib.nifti1.Nifti1Image(image_data, affine=nib_image.affine, header=nib_image.header)
