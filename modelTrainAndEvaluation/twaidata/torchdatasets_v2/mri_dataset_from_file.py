import os
import SimpleITK as sitk
from torch.utils.data import Dataset
from collections import defaultdict
from natsort import natsorted
import numpy as np
import torch
from twaidata.mri_dataset_directory_parsers.generic import DirectoryParser

class MRISegmentationDatasetFromFile(Dataset):
    def __init__(self, dataset_parser:DirectoryParser, domain_key=None, extra_filetypes=["mask"], csv_path=None):
        
        """
        Creates a 3D MRI torch dataset
         
        dataset_parser object that specifies that output paths and files of a specific dataset.
                    
        domain_key: optional key matching a subset of files. Must match a domain key known by the parser.
        
        filetypes_subset
        defined if only a subset of filetypes found in the imgs/ and labels/ folder are desired
        
        csv_path: optional path to a csv file that contains extra information for each individual
        
        extra_filetypes: these are files that are found in the relevant imgs or labels folder for each individual that are unknown by the parser (because they are derived data that are produced in extra processing). They should be of the format <ind>_<filetype>.nii.gz where <filetype> is in the extra_filetypes list.
        
        """
        
        self.iomap = dataset_parser.get_dataset_inout_map()
        keys = list(self.iomap.keys())
    
        if domain_key:
            domain_map = dataset_parser.get_domain_map()
            keys = [key for key in keys if domain_map[key] == domain_key]
            
        self.keys = natsorted(keys)
        
        self.extra_filetypes = None
        if extra_filetypes:
            self.extra_filetypes = natsorted(extra_filetypes)
        
        ### deal with the csv file if it is there
        self.csv_data = None
        if csv_path:
            df = pd.read_csv(csv_path)
            filtered = df.loc[df['ID'].isin(self.keys)]
            self.csv_data = filtered
        
    def _load_image(self, file, file_key, islabel, xs, ys):
        if not file.endswith(".nii.gz"):
            file = file + ".nii.gz"
        
        img = sitk.GetArrayFromImage(sitk.ReadImage(file)).astype(np.float32)
        img = torch.from_numpy(img)

        if "ICV" in file_key:
            file_key = "mask"
        
        if islabel and "imgs" not in file: # some labels are put in the images section (e.g mask, ventricle seg etc)
            ys[file_key] = img
        else:
            xs[file_key] = img
            
        
    def __getitem__(self, idx):
        key = self.keys[idx]
        individual_map = self.iomap[key]
        
        xs = {}
        ys = {}
        
        for file_key in natsorted(list(individual_map.keys())):
            # print(file_key)
            file_data = individual_map[file_key]
            outfile = os.path.join(file_data["outpath"], file_data["outfilename"])
            islabel = file_data["islabel"]
            
            self._load_image(outfile, file_key, islabel, xs, ys)
        
        
        if self.extra_filetypes:
            ds_data_path = "/".join(file_data['outpath'].split('/')[:-1])
            imgs_path = os.path.join(ds_data_path, "imgs")
            labels_path = os.path.join(ds_data_path, "labels")
            
            key_generic_file = file_data["outfilename"].replace(f"_{file_key}", "").replace(".nii.gz", "")
            # print(file_data["outfilename"])
            # print(key)
            # print(key_generic_file)
            
            for file_key in self.extra_filetypes:
                found = False
                for file in os.listdir(imgs_path):
                    if key_generic_file in file and file_key in file:
                        path = os.path.join(imgs_path, file)
                        islabel = False
                        found = True
                        break
                if found:
                    self._load_image(path, file_key, islabel, xs, ys)
                    continue
                for file in os.listdir(labels_path):
                    if key_generic_file in file and file_key in file:
                        path = os.path.join(imgs_path, file)
                        islabel = True
                        found = True
                        break
                if not found:
                    raise ValueError(f"could not find any match for extra filetype: {file_key} for individual: {key}")
                
                self._load_image(path, file_key, islabel, xs, ys)
                
        if self.csv_data is not None:
            return xs, ys, key, torch.from_numpy(self.csv_data.loc[self.csv_data['ID'] == key].iloc[0].values)
        
        return xs, ys, key, None
    
    def __len__(self):
        return len(self.keys)
        
        
class ArrayMRISegmentationDatasetFromFile(MRISegmentationDatasetFromFile):
    def __getitem__(self, idx):
        xs, ys, key, csv_data = super().__getitem__(idx)
        xs = torch.stack([xs['FLAIR'], xs['T1'], xs['mask']])
        ys = torch.stack(list(ys.values()))
        return xs, ys, key, csv_data
