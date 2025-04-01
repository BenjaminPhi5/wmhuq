from torch.utils.data import Dataset
from collections import defaultdict
from natsort import natsorted
import numpy as np
import torch
import os

class MRISegmentation2DDataset(Dataset):
    def __init__(self, imgs_dir, domain_name,
                 transforms=None):
        data_path = os.path.join(imgs_dir, domain_name)
        
        # load the data into ram cheeky but fine for smol data.
        # this dataset treates each slice as a separate training instance
        # assumed format is (n, c, d, h, w)
        # and so item i is at location, (i//D, : i - D * (i // D), :, :)
        # where D is the number of z slices in each image 
        
        self.imgs = torch.Tensor(np.load(os.path.join(data_path, "imgs.npy")))
        self.labels = torch.Tensor(np.load(os.path.join(data_path, "labels.npy")))
        self.dslices = self.imgs.shape[2]
        self.size = self.dslices * self.imgs.shape[0]
        
        self.transforms = transforms
            
    def __getitem__(self, idx):
        n = idx // self.dslices      
        d = idx - (self.dslices * n)
        img = self.imgs[n, :, d, :, :]
        label = self.labels[n, :, d, :, :]
        
        if self.transforms:
            img, label = self.transforms(img, label)
            
        return img, label

    def __len__(self):
        return self.size
    
    
class MRISegmentation3DDataset(Dataset):
    """
    stores a whole dataset in memory
    loads dataset from a pyton file.
    """
    def __init__(self, imgs_dir, domain_name,
                 transforms=None):
        data_path = os.path.join(imgs_dir, domain_name)
        
        self.imgs = torch.Tensor(np.load(os.path.join(data_path, "imgs.npy")))
        self.labels = torch.Tensor(np.load(os.path.join(data_path, "labels.npy")))
        self.dslices = self.imgs.shape[2]
        self.size = self.imgs.shape[0]
        
        self.transforms = transforms
            
    def __getitem__(self, idx):
        img = self.imgs[idx]
        label = self.labels[idx]
        
        if self.transforms:
            img, label = self.transforms(img, label)
            
        return img, label

    def __len__(self):
        return self.size