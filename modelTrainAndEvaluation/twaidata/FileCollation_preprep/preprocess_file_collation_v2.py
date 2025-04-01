"""

takes all the preprocessed files for a given domain and collates them into one numpy array file.
This way an entire dataset can be loaded into memory and retained, much less file IO during training.
This is version 2 that makes use of the fileparsers to build the collated dataset and works with the new code.
"""

import numpy as np
from twaidata.torchdatasets_v2.mri_dataset_from_file import MRISegmentationDatasetFromFile
from twaidata.mri_dataset_directory_parsers.parser_selector import select_parser
import torch
import os
from pathlib import Path
from trustworthai.utils.augmentation.standard_transforms import NormalizeImg, PairedCompose, LabelSelect, PairedCentreCrop, CropZDim
import argparse
from natsort import natsorted
from tqdm import tqdm

def construct_parser():
    # preprocessing settings
    parser = argparse.ArgumentParser(description = "MRI nii.gz simple preprocessing pipeline")
    
    parser.add_argument('-i', '--in_dir', required=True, help='Path to parent of the dataset to be preprocessed')
    parser.add_argument('-o', '--out_dir', required=True, help='Path to the preprocessed data folder')
    parser.add_argument('-c', '--csv_file', default=None, help='CSV file containing preprocessing data for custom datasets')
    parser.add_argument('-g', '--data_csv_file', default=None, help='data CSV file containing information about each individual, e.g age, fazekas, clinical data etc')
    parser.add_argument('-n', '--name', required=True, help='Name of dataset to be processed')
    parser.add_argument('-d', '--domain_key', required=False, default=None, help="Subdomain of the dataset to be processed")
    parser.add_argument('-e', '--extra_filetypes', required=False, default='mask', help='list of any extra image modalities / masks that are not included by the fileparser')
    parser.add_argument('-a', '--add_dsname_to_folder_name', default="False", type=str)
    parser.add_argument('-H', '--crop_height', required=True, default=224, type=int, help="height of the centre crop of the image")
    parser.add_argument('-W', '--crop_width', required=True, default=160, type=int, help="width of the centre crop of the image")
    parser.add_argument('-l', '--label_extract', required=False, default=None, type=int, help="specfic id in the label map to extract (e.g 1 is WMH, 2 is other pathology in the WMH challenge dataset. if set, only the given label will be extracted, otherwise the label will be left as is). optional")

    return parser


def main(args):
    # extract args
    in_dir = args.in_dir
    out_dir = args.out_dir
    name = args.name
    domain = args.domain_key
    crop_height = args.crop_height
    crop_width = args.crop_width
    label_extract = args.label_extract
    
    # check file paths are okay     
    if not os.path.exists(in_dir):
        raise ValueError(f"could not find folder: {in_dir}")
            
    
    # select centre crop and optionaly label extract transform
    crop_size = (crop_height, crop_width)
    transforms = get_transforms(crop_size, label_extract)
    
    # setup folder for collation outputs
    collation_out_dir = os.path.join(out_dir, "collated")
    if not os.path.exists(collation_out_dir):
        os.makedirs(collation_out_dir)
    filename = "collated"
    if args.domain_key:
        filename = args.domain_key + "_" + filename
    outpath = os.path.join(collation_out_dir, filename)
            
    # get the parser that maps inputs to outputs
    # csv file used for custom datasets
    parser = select_parser(args.name, args.in_dir, args.out_dir, args.csv_file, args.add_dsname_to_folder_name.lower() == "true")
            
    # load the dataset
    print(f"loading dataset {args.name} for domain split {args.domain_key}")
    extra_filetypes = None if args.extra_filetypes.lower() == "none" else args.extra_filetypes
    extra_filetypes = extra_filetypes.split(",") if extra_filetypes is not None else None
    dataset = MRISegmentationDatasetFromFile(
        dataset_parser=parser, domain_key=args.domain_key, extra_filetypes=extra_filetypes , csv_path=None
    )

    # collect the images and labels in to a list
    data_xs = []
    data_ys = []
    data_x_keys = []
    data_y_keys = []
    ids = []
    csv_datas = []
    slices = [] # check for inconsistent slice sizes across a domain
    for xs, ys, ind, csvd in tqdm(dataset):
        x_keys = natsorted(list(xs.keys()))
        x_arr = torch.stack([xs[key] for key in x_keys], dim=0)
        y_keys = natsorted(list(ys.keys()))
        if len(y_keys) > 0:
            y_arr = torch.stack([ys[key] for key in y_keys], dim=0)
        else:
            y_arr = x_arr
        
        x_arr, y_arr = transforms(x_arr, y_arr)
        
        if len(y_keys) == 0:
            y_arr = None
        
        data_xs.append(x_arr)
        data_ys.append(y_arr)
        data_x_keys.append(x_keys)
        data_y_keys.append(y_keys)
        ids.append(ind)
        csv_datas.append(csvd)
        slices.append(x_arr.shape[1])
        
    # where there is more than one slice size in the domain
    # take a centre crop of the sizes equal to the miniumum
    # number of slices found in the domain.
    # should not affect the WMH challenge data, only the ED inhouse data.
    print("calculating minimum number of slices found in the dataset")
    slices = np.array(slices)
    uniques = np.unique(slices)
    if len(uniques) > 1:
        print(f"unique slice sizes found in domain: {uniques}")
        # for each image select the centre minimum slice
        centre_cut = np.min(slices)
        for i in range(len(data_xs)):
            if centre_cut < data_xs[i].shape[1]: # crop images larger than the biggest slice size.
                start = (data_xs[i].shape[1] - centre_cut) // 2
                data_xs[i] = data_xs[i][:,start:start+centre_cut,:,:]
                if data_ys[i] is not None:
                    data_ys[i] = data_ys[i][:,start:start+centre_cut,:,:]
        
    # convert to a single numpy array for the data elements
    print("converting all data to numpy")
    data_xs = torch.stack(data_xs, dim=0).numpy()
    if data_ys[0] is not None:
        data_ys = torch.stack(data_ys, dim=0).numpy()
    if csv_datas[0] is not None:
        csv_datas = torch.stack(csv_datas, dim=0).numpy()
    ids = np.array(ids)
    data_x_keys = np.array(data_x_keys)
    data_y_keys = np.array(data_y_keys)
    
    print(f"dataset imgs shape: {data_xs.shape}")
    if data_ys[0] is not None:
        print(f"dataset labels shape: {data_ys.shape}") 

    # save the file
    np.savez(outpath, xs=data_xs, ys=data_ys, csv_datas=csv_datas, ids=ids, data_x_keys=data_x_keys, data_y_keys=data_y_keys)

            
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

