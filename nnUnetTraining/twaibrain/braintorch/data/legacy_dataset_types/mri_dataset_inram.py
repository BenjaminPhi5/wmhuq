"""
for now v2 version very similar to v1 version. However, it has accomodation for storing ids,
the names of image modalities and labels, along with extra csv data (age, fazekas) etc if it is stored
in a spreadsheet file. I can implement that for later.
Uses the new version of the data preprocessing pipeline that I have built.
"""


from torch.utils.data import Dataset
from collections import defaultdict
from natsort import natsorted
import numpy as np
import torch
import os
import pandas as pd

class MRISegmentation2DDataset(Dataset):
    def __init__(self, imgs_dir, domain_name=None,
                 transforms=None, no_labels=False, xy_only=False, label_key=None, label_value=None):
        
        # construct path to collated data file
        if domain_name is None:
            data_path = os.path.join(imgs_dir, "collated.npz")
        else:
            data_path = os.path.join(imgs_dir, f"{domain_name}_collated.npz")
        
        self.xy_only = xy_only # whether the model only returns x and y, or whether it returns all info
        self.no_labels = no_labels # whether there are labels for this dataset or not
        
        # load the data into ram cheeky but fine for smol data.
        arr = np.load(data_path, allow_pickle=True)
        xs, ys, csv_datas, ids, data_x_keys, data_y_keys = arr['xs'], arr['ys'], arr['csv_datas'], arr['ids'], arr['data_x_keys'], arr['data_y_keys']
        
        # this dataset treates each slice as a separate training instance
        # assumed format is (n, c, d, h, w)
        # and so item i is at location, (i//D, : i - D * (i // D), :, :)
        # where D is the number of z slices in each image 
        self.imgs = torch.from_numpy(xs)
        if no_labels:
            self.labels = [None for _ in range(len(xs))]
        else:
            try:
                self.labels = torch.from_numpy(ys)
            except:
                self.labels = ys
        self.data_x_keys = data_x_keys
        self.data_y_keys = data_y_keys
        
        self.csv_datas = csv_datas
        if self.csv_datas[0] is not None:
            self.csv_datas = torch.from_numpy(self.csv_datas)
        
        if not self.xy_only:
            self.ids = ids
            self.data_x_keys = data_x_keys
            self.data_y_keys = data_y_keys
        
        self.dslices = self.imgs.shape[2]
        self.size = self.dslices * self.imgs.shape[0]
        
        self.transforms = transforms
        
        self.do_filter_labels = False
        if label_key is not None:
            self.do_filter_labels = True
            self.label_key = label_key
            self.label_value = label_value
            
    def _filter_labels(self, label, keys):
            selected_idx = None
            for i, key in enumerate(keys):
                if key == self.label_key:
                    selected_idx = i
                    break
            
            label = label[selected_idx]
            if self.label_value is not None:
                label = (label == self.label_value).type(label.dtype)
                
            return label
            
    def __getitem__(self, idx):
        n = idx // self.dslices      
        d = idx - (self.dslices * n)
        img = self.imgs[n, :, d, :, :]
        label = self.labels[n, :, d, :, :]
        
        if self.do_filter_labels:
            label = self._filter_labels(label, self.data_y_keys[n])
            
        if self.transforms:
            img, label = self.transforms(img, label)
            
        if self.xy_only:
            return img, label
        else:
            return img, label, self.ids[n], self.csv_datas[n], self.data_x_keys[n], self.data_y_keys[n]
        
    def __len__(self):
        return self.size
    
    
class MRISegmentation3DDataset(MRISegmentation2DDataset):
    """
    stores a whole dataset in memory
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.size = self.imgs.shape[0]
            
    def __getitem__(self, idx):
        img = self.imgs[idx]
        label = self.labels[idx]
        
        if self.do_filter_labels:
            label = self._filter_labels(label, self.data_y_keys[idx])
        
        if self.transforms:
            img, label = self.transforms(img, label)
            
        if self.xy_only:
            return img, label
        else:
            return img, label, self.ids[idx], self.csv_datas[idx], self.data_x_keys[idx], self.data_y_keys[idx]

    def __len__(self):
        return self.size

class CSVDataWrapper(Dataset):
    """
    replaces the csv data index from the in-memory dataset with a post-hoc csv.
    Assumes that the patient id will match the id from the csv file.
    """
    def __init__(self, base_dataset:MRISegmentation2DDataset, csv_file:str, id_key="Patient ID"):
        df = pd.read_csv(csv_file)
        self.base_dataset = base_dataset
        self.df = df
        self.id_key = id_key
        
    def __getitem__(self, idx):
        img, label, patient_id, _, xkeys, ykeys = self.base_dataset[idx]
        
        csv_data = self.df[self.df[self.id_key]==patient_id].values
        
        return img, label, patient_id, csv_data, xkeys, ykeys
    
    def __len__(self):
        return len(self.base_dataset)
