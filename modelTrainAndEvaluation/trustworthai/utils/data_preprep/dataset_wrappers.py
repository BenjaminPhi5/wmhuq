"""
a collection of torch dataset wrappers that I used for CNN models for predicting Fazekas
"""

# generated partly using chat-gpt.
import torch
import random
from torch.utils.data import Dataset
import numpy as np
import os
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from natsort import natsorted

class CustomClinFieldsDataset(torch.utils.data.Dataset):
    # this is just a wrapper on the clinical dataset where I am hard coding a few things, such as one hot encodings, and all the other stuff that
    # I might want to compute like tab fazekas and total fazekas etc
    def __init__(self, base_dataset):
        super().__init__()
        self.base_dataset = base_dataset
        
    def __getitem__(self, idx):
        x, y, clin_data = self.base_dataset[idx]
        r = clin_data.copy()
        
        # smoking
        r['smoking_0'] = (r['smoking'] == 0).astype(np.float32)
        r['smoking_1'] = (r['smoking'] == 1).astype(np.float32)
        r['smoking_2'] = (r['smoking'] == 2).astype(np.float32)
        
        # total fazekas and scale fazekas
        dwmh = r['DWMH']
        pvwmh = r['PVWMH']
        total_fazekas = np.nan
        if (not np.isnan(dwmh)) and (not np.isnan(pvwmh)):
            total_fazekas = dwmh + pvwmh
        r['total_fazekas'] = total_fazekas
        r['scale_fazekas'] = ((pvwmh == 3) | (dwmh > 1)).astype(np.float32)
        
        # scale PVS
        bgpvs = r['BGPVS']
        scale_pvs = np.nan
        if not np.isnan(bgpvs):
            scale_pvs = (bgpvs >= 2).astype(np.float32)
        r['scale_pvs'] = scale_pvs
        
        # scale micrBld
        micrbld = r['micrBld']
        scale_micrbld = np.nan
        if not np.isnan(scale_micrbld):
            scale_micrbld = (micrbld > 0).astype(np.float32)
        r['scale_micrbld'] = scale_micrbld
        
        # stroke_les and scale stroke
        oldLes = r['oldLes']
        relLes = r['relLes']
        stroke_les = np.nan
        scale_stroke = np.nan
        if not np.isnan(oldLes):
            if type(relLes) == str:
                if relLes != ' ':
                    relLes = float(relLes)
                else:
                    relLes = 0.0
            if type(oldLes) == str:
                if oldLes != ' ':
                    oldLes = float(oldLes)
                else:
                    oldLes = 0.0
            try:
                if np.isnan(relLes):
                    relLes = 0.0
            except:
                print("failed on : ", relLes, type(relLes))
            
            try:
                stroke_les = ((oldLes ==1)| (relLes==1)).astype(np.float32)
            except:
                print(f"failed: old:{oldLes}, rel:{relLes}")
            scale_stroke = (oldLes * relLes).astype(np.float32)
        r['stroke_les'] = stroke_les
        r['scale_stroke'] = scale_stroke
            
        return x, y, r
    
    def __len__(self):
        return len(self.base_dataset)
    
    
class AddChannelsDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset, extra_x_channels_lists):
        self.base_dataset = base_dataset
        self.extra_x_channels_lists = extra_x_channels_lists
        
    def __getitem__(self, idx):
        data = self.base_dataset[idx]
        x = data[0]
        x = torch.cat([x, *[self.extra_x_channels_lists[key][idx].unsqueeze(0) for key in natsorted(self.extra_x_channels_lists.keys())]], dim=0)
        
        return (x, *data[1:])
        
    def __len__(self):
        return len(self.base_dataset)
    

    
### code to select slices - will need to do that randomly, and implement an augmentation procedure (I should add noise maybe)

# a bunch of stuff that I generated with chat-gpt

def find_largest_slice(tensor):
    """
    Find the slice of a PyTorch tensor with the largest sum.
    
    Args:
        tensor: A PyTorch tensor of shape (S, H, W).
        
    Returns:
        A tuple (slice_H, slice_W) representing the slice with the largest sum.
    """
    # Compute the sum along the H and W dimensions
    sum_H = torch.sum(tensor, dim=2)
    sum_W = torch.sum(tensor, dim=1)
    
    # Find the indices with the largest sum
    idx_H = torch.argmax(sum_H)
    idx_W = torch.argmax(sum_W)
    
    # Return the corresponding slices
    return idx_H, idx_W

def compute_std_of_sum(tensor):
    """
    Compute the standard deviation of the sum along H and W dimensions for the given image.
    
    Args:
        tensor: A PyTorch tensor of shape (S, H, W).
        
    Returns:
        A float of standard deviation of the sum along H and W dimensions for the given image.
    """
    # Compute the sum along the H and W dimensions
    sum_HW = torch.sum(tensor, dim=(1, 2))
    
    # Compute the standard deviation of the sum along the H and W dimensions
    #print(tensor.shape, sum_HW.shape)
    std_HW = torch.std(sum_HW, dim=0)
    
    return std_HW.item()

def find_slices_within_std(tensor, t=1.0):
    """
    Find the slices whose sum are within t standard deviations away from the max slice sum for the given image.
    
    Args:
        tensor: A PyTorch tensor of shape (S, H, W).
        t: A float representing the number of standard deviations away from the max slice sum to include.
        
    Returns:
        A list containing the indices of the slices whose sum is within t standard deviations away from the max slice sum in the given image.
    """
    # Compute the sum along the H and W dimensions
    sum_HW = torch.sum(tensor, dim=(1, 2))
    
    # Compute the standard deviation of the sum for each image
    std_sum = compute_std_of_sum(tensor)
    
    # Compute the max slice sum for the image
    max_sum = torch.max(sum_HW, dim=0)[0]
    
    # Compute the threshold for including a slice
    threshold = max_sum - t * std_sum
    
    # Find the slices whose sum is within the threshold for each image
    indices_within_std = torch.where(sum_HW >= threshold)[0].tolist()
    
    if len(indices_within_std) < 3: # edge cases where too few slices are selected.
        indices_within_std += [indices_within_std[-1] + 1, indices_within_std[0] - 1]
    
    return indices_within_std

class SlicesDataset(Dataset):
    """
    A PyTorch Dataset that selects v slices from 3D images within the range of standard deviations of the max slice sum.
    """
    
    def __init__(self, base_dataset, slices_within_std, v, transform=None):
        """
        Initialize the SlicesDataset.
        
        Args:
            base_dataset: A PyTorch Dataset that returns 3D images of shape (C, S, H, W).
            slices_within_std: A list of N lists, where the i-th list contains the indices of the slices whose sum is within t standard deviations away from the max slice sum for the i-th image in the batch.
            v: An integer representing the number of slices to select from each image.
            transform: A function to be applied to both x and y when returning a value from __getitem__. Default is None.
        """
        self.base_dataset = base_dataset
        self.slices_within_std = slices_within_std
        self.v = v
        self.transform = transform
    
    def __getitem__(self, index):
        """
        Select v random slices from the 3D image within the range of standard deviations, and return a tensor of shape (C * v, H, W).
        
        Args:
            index: An integer representing the index of the image to retrieve.
            
        Returns:
            A tuple containing a PyTorch tensor of shape (C * v, H, W) and its corresponding label.
        """
        # Get the original 3D image and its corresponding label
        x_3d, y_3d, clin_data = self.base_dataset[index]
        
        # Get the indices of the slices within the range of standard deviations
        indices_within_std = self.slices_within_std[index]
        
        # Randomly select v slices from the indices within the range of standard deviations
        selected_indices = random.sample(indices_within_std, self.v)
        
        # Select the slices from the 3D image
        x_slices = x_3d[:, selected_indices]
        y_slices = y_3d[:, selected_indices]
        
        # Reshape the slices into C*v 2D tensors
        x_2d = torch.reshape(x_slices, (-1, x_slices.shape[-2], x_slices.shape[-1]))
        y_2d = torch.reshape(y_slices, (-1, y_slices.shape[-2], y_slices.shape[-1]))
        
        # Apply the transform function to both x and y
        if self.transform:
            x_2d, y_2d = self.transform(x_2d, y_2d)
        
        return x_2d, y_2d, clin_data
    
    def __len__(self):
        """
        Return the number of images in the dataset.
        
        Returns:
            An integer representing the number of images in the dataset.
        """
        return len(self.base_dataset)
    
    
def get_slices_within_std_for_ds(ds, t=2.5):
    # get slices with wmh sum within 3 std of the maximum
    # I could also replace with for slices within 3 std of the max uncertainty?
    slices_within_std = []
    for data in tqdm(ds, position=0, leave=True):
        y = data[1].squeeze()
        slices_within_std.append(find_slices_within_std(y, t=t))
        
    return slices_within_std


### combine the clinical scores data into the x information.
# generated with chatgpt
class ClinicalDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset, fields, target_field):
        self.base_dataset = base_dataset
        self.fields = fields
        self.target_field = target_field

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index):
        x, y, clin_data = self.base_dataset[index]
        clin_data_fields = clin_data[self.fields].values
        clin_data_tensor = torch.from_numpy(clin_data_fields.astype(np.float32))
        target_field = clin_data[self.target_field]
        return (x, clin_data_tensor), target_field
    

    # torch dataset that filters out nans
class NonNanDataset(Dataset):
    def __init__(self, original_dataset):
        self.original_dataset = original_dataset
        self.indices = []

        for i in range(len(self.original_dataset)):
            (x, clin_data), y = self.original_dataset[i]
            if not (np.isnan(y) or torch.any(torch.isnan(clin_data))):
                self.indices.append(i)

    def __getitem__(self, index):
        original_index = self.indices[index]
        return self.original_dataset[original_index]

    def __len__(self):
        return len(self.indices)
    
class RepeatDataset(Dataset):
    def __init__(self, original_dataset, repeats):
        self.original_dataset = original_dataset
        self.repeats=repeats
        
    def __getitem__(self, idx):
        return self.original_dataset[idx % len(self.original_dataset)]
    
    def __len__(self):
        return len(self.original_dataset) * self.repeats

