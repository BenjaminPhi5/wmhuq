import SimpleITK as sitk
import os

def save_manipulated_sitk_image_array(source_image, target_array, filepath):
    """Saves a manipulated nifti image array using the meta data information from a source image"""
    target_image = sitk.GetImageFromArray(target_array)
    target_image.SetSpacing(source_image.GetSpacing())
    target_image.SetOrigin(source_image.GetOrigin())
    target_image.SetDirection(source_image.GetDirection())
    target_image.CopyInformation(source_image)
    sitk.WriteImage(target_image, filepath)

def load_image(filepath):
    return sitk.GetArrayFromImage(sitk.ReadImage(filepath))
