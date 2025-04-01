import os
import nibabel as nib
import SimpleITK as sitk
from torch.utils.data import Dataset
from collections import defaultdict
from natsort import natsorted
import numpy as np
import torch


class MRISegmentationDatasetFromFile(Dataset):
    def __init__(self, imgs_dir, 
                 img_filetypes=["FLAIR_BET_mask.nii.gz", "FLAIR.nii.gz", "T1.nii.gz"], # brain mask, flair, T1.
                 label_filetype="wmh.nii.gz",
                 transforms=None):
        
        # locate each individual, each file should be formatted
        # individual_filetype.fileending
        # if domains == None:
        # domains logic has been commented out because actually
        # we probably want to control how much of each domain we use,
        # not just do it randomly to ensure we get a balanced split.
        # so its better if the domains are combined later and separately.
        imgs_dir_files = os.listdir(os.path.join(imgs_dir, "imgs"))
        target_dir_files = os.listdir(os.path.join(imgs_dir, "labels"))
        # else:
        #     imgs_dir_files = []
        #     target_dir_files = []
        #     for d in domains:
        #         imgs_dir_files.append(os.listdir(os.path.join(imgs_dir, "imgs")))
        #         target_dir_files.append(os.listdir(os.path.join(imgs_dir, "labels")))
        
        # identify each individual
        individuals_map = defaultdict(lambda : {})
        for f in imgs_dir_files:
            try:
                ind = f.split("_")[0]
                filetype = f.split(f"{ind}_")[1]
                if filetype not in img_filetypes:
                    continue # i.e ignore files in the folder that don't match target names exactly (e.g BET files, normalize files etc
            except:
                print(f"file {f} could not be parsed, skipping...")
            
            individuals_map[ind][filetype] = f
            
        # check that each individual has the same number of keys
        expected_keys = len(img_filetypes)
        for ind in individuals_map.keys():
            if len(individuals_map[ind].keys()) != expected_keys:
                raise ValueError(f"for individual {ind} expected exactly the following filetypes: "
                                 f"'{ind}_' + '{img_filetypes}' BUT only {list(individuals_map[ind].values())} were found")
                
        # check that each individual has a label
        individuals_labels_map = {}
        for ind in individuals_map.keys():
            label_file = f"{ind}_{label_filetype}"
            if label_file not in target_dir_files:
                raise FileNotFoundError(f"Could not find label {label_file} ")
            else:
                individuals_labels_map[ind] = label_file
        
        # initialize object
        self.imgs_dir = imgs_dir
        self.individuals = natsorted(list(individuals_map.keys()))
        self.img_filetypes = natsorted(list(img_filetypes)) # ensure it is a fixed ordered object
        self.label_filetype = label_filetype
        self.transforms = transforms
            
    def __getitem__(self, idx):
        # compute paths to images and labels
        img_paths = [
            os.path.join(*[self.imgs_dir, "imgs", f"{self.individuals[idx]}_{filetype}"])
            for filetype in self.img_filetypes
        ]
        label_path = os.path.join(*[self.imgs_dir, "labels", f"{self.individuals[idx]}_{self.label_filetype}"])

        # loading brain scans
        # nib version loads depth as the last dimension
        # but torch wants it the first (it wants (n, c, d, h, w)
        # which sitk below gives
        # images = [nib.load(img_path) for img_path in img_paths]
        # images = [img.get_fdata().astype(np.float32) for img in images]
        # image = np.concatenate(images)
        # label = nib.load(label_path).get_fdata().astype(np.float32)
        
        # sitk equivalent
        images = [sitk.ReadImage(img_path) for img_path in img_paths]
        images = [sitk.GetArrayFromImage(img) for img in images]
        images = [img.astype(np.float32) for img in images]
        label = sitk.ReadImage(label_path)
        label = sitk.GetArrayFromImage(label)
        label = label.astype(np.float32)
        
        # convert to tensors
        if len(images) == 1:
            image = torch.FloatTensor(images[0])
        else:
            try:
                images = np.stack(images, axis=0)
            except:
                print("\n\nFAILED")
                for i in images:
                    print(i.shape)
                print("img_paths: ", img_paths)
                print("\n\n")
                raise ValueError("thigns went wrong")
            image = torch.FloatTensor(images)
        label = torch.FloatTensor(label).unsqueeze(0) # nesseary to match the dimensions of the images

        # apply transforms
        if self.transforms:
            image, label = self.transforms(image, label)

        return image, label

    def __len__(self):
        return len(self.individuals)
        