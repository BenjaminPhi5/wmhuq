import torch

def normalize_brain(image, mask, lower_percentile=0, upper_percentile=100, verbose=False):
    """
    function to normalize the brain within the mask area
    """
    
    # image = image.copy()
    mask = (mask==1)
    brain_locs = image[mask]
    
    if lower_percentile > 0 or upper_percentile < 100:
        if verbose:
            print("normalizing with percentiles: ", lower_percentile, upper_percentile)
        
        lower_percentile /= 100
        upper_percentile /= 100
        
        brain_locs = brain_locs.flatten()
        sorted_indices = torch.argsort(brain_locs)
        num_brain_voxels = len(sorted_indices)

        lower_index = int(lower_percentile*num_brain_voxels)
        upper_index = int(upper_percentile*num_brain_voxels)

        retained_indices = sorted_indices[lower_index:upper_index]
        
        brain_locs = brain_locs[lower_index:upper_index]
    else:
        if verbose:
            print("no percentiles used for normalization")

    mean = brain_locs.mean()
    std = brain_locs.std()
    
    print(mean, std)

    image[mask] = (image[mask] - mean) / std

    return image
