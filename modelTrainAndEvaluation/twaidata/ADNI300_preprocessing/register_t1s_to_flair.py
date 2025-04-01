import os
from twaidata.MRI_preprep.register_two_images import register_two_images
from tqdm import tqdm
import SimpleITK as sitk

def register_all_t1_to_flair(imgs_folder, force_individuals=False):
    # imgs_folder is a folder containing a folder per each ID-date pair.
    # inside each of those folders should be a t1w.nii.gz and flair.nii.gz image. 
    # this method loops over each folder, registers the t1 to the flair and puts
    # a t1w_registered.nii.gz in each folder. Nice.
    
    subfolders = os.listdir(imgs_folder)
    
    for subfolder in tqdm(subfolders, position=0, leave=True):
        folder_path = os.path.join(imgs_folder, subfolder)
        t1_registered_path = os.path.join(folder_path, "t1w_registered.nii.gz")
        if not force_individuals and os.path.exists(t1_registered_path):
            continue
        
        flair_path = os.path.join(folder_path, "flair.nii.gz")
        t1_path = os.path.join(folder_path, "t1w.nii.gz")
        
        _, registered_t1 = register_two_images(flair_path, t1_path)
        
        sitk.WriteImage(registered_t1, t1_registered_path)
