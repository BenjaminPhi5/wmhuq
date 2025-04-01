"""
scans the entire subdirectory and runs synthseg on all files that end in a list of given file endings. Nice.
"""

import numpy as np
import nibabel as nib
import SimpleITK as sitk
import os
import subprocess
import torch
from scipy.ndimage import distance_transform_edt
from twaidata.MRI_preprep.io import save_nii_img, load_nii_img
from twaidata.MRI_preprep.resample import resample_images_to_original_spacing_and_save
import argparse
from natsort import natsorted
from pathlib import Path
from tqdm import tqdm


SYNTH_SEG_PYTHON_PATH = "/home/s2208943/miniconda3/envs/synthseg_38/bin/python"
SYNTH_SEG_PREDICT_PATH = "/home/s2208943/SynthSeg/scripts/commands/SynthSeg_predict.py"

VENTRICLE_1 = 4
VENTRICLE_2 = 43

def run_synthseg(in_file, out_folder, threads=6):
    """
    runs synth seg on in_file and saves the result in out_folder
    """
    subprocess.run([
        SYNTH_SEG_PYTHON_PATH,
        SYNTH_SEG_PREDICT_PATH,
        "--i", in_file,
        "--o", out_folder, 
        "--threads=6",
    ])

def create_ventricle_distance_map(synthseg_file, out_file):
    """
    loads the synth_seg segmentation, extracts the ventricles segmentation
    and creates a euclidian distance map from each voxel to the ventricles.
    This distance map is then saved under the name out_file
    """
    
    synthseg, synthseg_header = load_nii_img(synthseg_file)
    spacing = sitk.ReadImage(synthseg_file).GetSpacing()
    if spacing[0] != 1 or spacing[1] != 1 or spacing[2] != 1:
        raise ValueError(f"image spacing must be (1, 1, 1) to compute distance map, not {spacing}")
    
    ventricle_seg = ((synthseg == VENTRICLE_1) | (synthseg == VENTRICLE_2)).astype(np.float32)
    distance_map = distance_transform_edt(1 - ventricle_seg)
    save_nii_img(out_file, distance_map, synthseg_header)
    
    
def set_synth_seg_images_to_correct_shape(orig_image_path, new_image_paths, dtypes):
    """
    since the resampling of the synth seg images sometimes adds an extra slice, we just crop the synth seg segmentation to the right size. This is also applied to derived images (e.g the eucidean distance map).
    """
    
    orig_shape = load_nii_img(orig_image_path)[0].shape  
    
    # load, crop, update header, save
    for path, dtype in zip(new_image_paths, dtypes):
        data, header = load_nii_img(path)
        data = data[0:orig_shape[0], 0:orig_shape[1], 0:orig_shape[2]]
        data = data.astype(dtype)
        header.set_data_dtype(dtype)
        save_nii_img(path, data, header)
        
def run_ventricle_seg_pipeline(in_file, out_folder, force=False):
    # compute names of files we need
    filename = in_file.split(".nii.gz")[0]
    synthseg_file = filename + "_synthseg.nii.gz"
    ventdistance_file = filename + "_vent_distance.nii.gz"
    
    # skip if file exists
    if not force and os.path.exists(ventdistance_file):
        print(f"SKIPPING {in_file} since file exists and force=False")
        return
    
    print(f"PROCESSING {in_file}")
    Path(ventdistance_file).touch() # create this so that other processes running in parallel
    # dont start working on the same file.
    
    
    # run synth seg
    run_synthseg(in_file, out_folder)
    
    # create ventricle distance map
    create_ventricle_distance_map(synthseg_file, ventdistance_file)
    
    # resample the synthseg and ventricle distance map
    resample_images_to_original_spacing_and_save(
        orig_image_path=in_file,
        new_image_paths=[synthseg_file, ventdistance_file],
        is_labels=[True, False],
    )
    
    # adjust the size of the synthseg and ventricle distance map to
    # be the right shape after resampling
    set_synth_seg_images_to_correct_shape(orig_image_path=in_file,
        new_image_paths=[synthseg_file, ventdistance_file], dtypes=[np.uint8, np.float32])


def run_vent_pipline_across_folder(folder, filetypes, force=False):
    matching_files = []
    print("filetypes: ", filetypes)
    for root, dirs, files in os.walk(folder):
        for file in files:
            for filetype in filetypes:
                if file.endswith(f"_{filetype}.nii.gz"):
                    matching_files.append((root, os.path.join(root, file)))
                    break
    
    print("num files found: ", len(matching_files))
    
    # run pipeline
    for root, f in matching_files:
        run_ventricle_seg_pipeline(f, root, force=force)


def construct_parser():
    # preprocessing settings
    parser = argparse.ArgumentParser(description = "MRI nii.gz simple preprocessing pipeline")
    
    parser.add_argument('-i', '--in_dir', required=True, help='folder containing images ending _{<one of filetypes}.nii.gz to be processed', type=str)
    parser.add_argument('-t', '--filetypes', default="T1,ADC", type=str)
    parser.add_argument('-f', '--force', default="false", help='force apply', type=str)

    return parser

if __name__ == '__main__':
    parser = construct_parser()
    args = parser.parse_args()
    run_vent_pipline_across_folder(args.in_dir, args.filetypes.split(","), args.force.lower() == "true")
