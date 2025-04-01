"""

takes all the preprocessed files for a given domain and collates them into one numpy array file.
This way an entire dataset can be loaded into memory and retained, much less file IO during training.

"""

import numpy as np
from twaidata.torchdatasets.whole_brain_dataset import MRISegmentationDatasetFromFile
import torch
import os
from pathlib import Path
from trustworthai.utils.augmentation.standard_transforms import NormalizeImg, PairedCompose, LabelSelect, PairedCentreCrop, CropZDim
import argparse

def construct_parser():
    # preprocessing settings
    parser = argparse.ArgumentParser(description = "MRI nii.gz simple preprocessing pipeline")
    
    parser.add_argument('-i', '--in_dir', required=True, help='Path of the stage 1 preprocessed data input folder')
    parser.add_argument('-o', '--out_dir', required=True, help='Path of the stage 2 preprocessed data output folder')
    parser.add_argument('-n', '--name', required=True, help='Name of dataset to be processed')
    parser.add_argument('-d', '--domain', required=False, default=None, help="Subdomain of the dataset to be processed. If None, will search for data directly in in_dir/dataset_name")
    parser.add_argument('-H', '--crop_height', required=True, default=224, type=int, help="height of the centre crop of the image")
    parser.add_argument('-W', '--crop_width', required=True, default=160, type=int, help="width of the centre crop of the image")
    parser.add_argument('-l', '--label_extract', required=False, default=None, type=int, help="specfic id in the label map to extract (e.g 1 is WMH, 2 is other pathology in the WMH challenge dataset. if set, only the given label will be extracted, otherwise the label will be left as is). optional")

    return parser


def main(args):
    # extract args
    in_dir = args.in_dir
    out_dir = args.out_dir
    name = args.name
    domain = args.domain
    crop_height = args.crop_height
    crop_width = args.crop_width
    label_extract = args.label_extract
    
    # check file paths are okay
    in_dir = os.path.join(in_dir, name)
    out_dir = os.path.join(out_dir, name)
    if domain != None:
        in_dir = os.path.join(in_dir, domain)
        out_dir = os.path.join(out_dir, domain)
        
    if not os.path.exists(in_dir):
        raise ValueError(f"could not find folder: {in_dir}")
    
    print(f"processing dataset: {in_dir}")
    
    if not os.path.exists(out_dir):
        try:
            Path(out_dir).mkdir(parents=True, exist_ok=True)
        except FileNotFoundError:
            print(f"Warning: couldn't make output directory here: {out_dir}")
            
    
    # select centre crop and optionaly label extract transform
    crop_size = (crop_height, crop_width)
    transforms = get_transforms(crop_size, label_extract)
            
            
    # load the dataset
    dataset = MRISegmentationDatasetFromFile(
        in_dir, 
        img_filetypes=["FLAIR_BET_mask.nii.gz", "FLAIR.nii.gz", "T1.nii.gz"], # brain mask, flair, T1.
        label_filetype="wmh.nii.gz",
        transforms=transforms
    )

    # collect the images and labels in to a list
    data_imgs = []
    data_labels = []
    slices = [] # check for inconsistent slice sizes across a domain
    for (img, label) in dataset:
        data_imgs.append(img)
        data_labels.append(label)
        slices.append(img.shape[1])
        
    # where there is more than one slice size in the domain
    # take a centre crop of the sizes equal to the miniumum
    # number of slices found in the domain.
    # should not affect the WMH challenge data, only the ED inhouse data.
    slices = np.array(slices)
    uniques = np.unique(slices)
    if len(uniques) > 1:
        print(f"unique slice sizes found in domain: {uniques}")
        # for each image select the centre minimum slice
        centre_cut = np.min(slices)
        for i in range(len(data_imgs)):
            if centre_cut < data_imgs[i].shape[1]: # crop images larger than the biggest slice size.
                start = (data_imgs[i].shape[1] - centre_cut) // 2
                data_imgs[i] = data_imgs[i][:,start:start+centre_cut,:,:]
                data_labels[i] = data_labels[i][:,start:start+centre_cut,:,:]
        
    # convert to numpy arrays
    data_imgs = np.stack(data_imgs, axis=0)
    data_labels = np.stack(data_labels, axis=0)
    print(f"dataset imgs shape: {data_imgs.shape}") 
    print(f"dataset labels shape: {data_labels.shape}") 

    # save the files
    out_file_imgs = os.path.join(out_dir, "imgs.npy")
    out_file_labels = os.path.join(out_dir, "labels.npy")
    np.save(out_file_imgs, data_imgs)
    np.save(out_file_labels, data_labels)

            
def get_transforms(crop_size, label_extract):
    if label_extract == None:
        print("keeping all labels")
        return PairedCentreCrop(crop_size)
    else:
        print(f"extracting label {label_extract}")
        transforms = PairedCompose([
            PairedCentreCrop(crop_size),    # cut out the centre square
            LabelSelect(label_extract),     # extract the desired label
        ])
        return transforms

if __name__ == '__main__':
    parser = construct_parser()
    args = parser.parse_args()
    main(args)

