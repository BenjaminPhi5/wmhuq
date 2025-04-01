from wmhuq.preprocessing import get_resampled_img
from scipy.ndimage import distance_transform_edt
from wmhuq.preprocessing import OUT_SPACING
import SimpleITK as sitk
import numpy as np

VENTRICLE_1 = 4
VENTRICLE_2 = 43


def get_vent_dist(synthseg_img, verbose):
    pass

    # resample to 1x1x1 space
    spacing = synthseg_img.GetSpacing()
    if not ((0.95 <= spacing[0] <= 1.05) and (0.95 <= spacing[1] <= 1.05) and (0.95 <= spacing[2] <= 1.05)):
        synthseg_img = get_resampled_img(synthseg_img, [1., 1., 1.,], original_spacing=spacing, is_label=True, verbose=verbose)
        
    # calculate distance
    synthseg = sitk.GetArrayFromImage(synthseg_img)
    ventricle_seg = ((synthseg == VENTRICLE_1) | (synthseg == VENTRICLE_2)).astype(np.float32)
    distance_map = distance_transform_edt(1 - ventricle_seg)
            
    # resample back
    vent_dist_image = sitk.GetImageFromArray(distance_map)
    vent_dist_image.SetSpacing(synthseg_img.GetSpacing())
    vent_dist_image = get_resampled_img(vent_dist_image, OUT_SPACING, original_spacing=synthseg_img.GetSpacing(), is_label=True, verbose=verbose)
    
    return sitk.GetArrayFromImage(vent_dist_image)
