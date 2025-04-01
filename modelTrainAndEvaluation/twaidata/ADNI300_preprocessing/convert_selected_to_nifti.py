from twaidata.MRI_preprep.convert_to_nifti import convert_to_nifti_format
import os
from tqdm import tqdm

def convert_adni_folders_to_nifti(ID_flair_matches, ID_t1_matches, ID_date_matches, out_folder, force=False, force_individuals=False):
    # this function takes in the outputs from the adni flair and t1 matchers that crawl the adni folder and find id, date matching pairs.
    # it then for each of those identified flair and t1 images, converts them from analyze 7.5 to nifti using nibabel
    # and puts them in a new folder afterwards. not all ids that have a flair may have a matching t1 at the moment (due to different protocol?)
    # at the moment 2 of the 200 adni sample are skipped.
    if not os.path.exists(out_folder):
        os.mkdir(out_folder)
    else:
        if not force:
            print("folder already exists. stopping. set force to true to overwrite")
            return
        
    for ID, date in tqdm(ID_date_matches.items(), position=0, leave=True):
        flair_file = ID_flair_matches[ID]
        if ID in ID_t1_matches:
            t1w_file = ID_t1_matches[ID]
        else:
            continue
        
        subfolder = f"{out_folder}{ID}_{date}"
        if not os.path.exists(subfolder):
            os.mkdir(subfolder)
        
        flair_out = f"{subfolder}/flair.nii.gz"
        t1w_out = f"{subfolder}/t1w.nii.gz"
        
        if force_individuals or not os.path.exists(flair_out):
            convert_to_nifti_format(flair_file, flair_out)
        if force_individuals or not os.path.exists(t1w_out):
            convert_to_nifti_format(t1w_file, t1w_out)
