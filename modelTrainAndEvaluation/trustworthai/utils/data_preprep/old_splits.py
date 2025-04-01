"""
for computing train, val, test splits of a torch dataset
"""
import torch
from torch.utils.data import random_split, Dataset
import numpy as np

def train_val_test_split(dataset, val_prop, test_prop, seed):
        # I think the sklearn version might be prefereable for determinism and things
        # but that involves fiddling with the dataset implementation I think....
        size = len(dataset)
        test_size = int(test_prop*size) 
        val_size = int(val_prop*size)
        train_size = size - val_size - test_size
        train, val, test = random_split(dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(seed))
        return train, val, test
    
    
def cross_validate_split(dataset, val_prop, test_prop, seed, split, test_fold_smooth=1):
    size = len(dataset)
    #test_size = int(np.round(test_prop*size)) + test_fold_smooth
    test_size = int(test_prop*size)
    val_size = int(val_prop*size)
    train_size = size - val_size
    indexes = np.arange(0, size, 1)
    rng = np.random.default_rng(seed)
    rng.shuffle(indexes)
    
    split_size = test_size
    assert split <= (size // split_size) - 1
    #assert split <= int(1 // test_prop) - 1
    

    if split == 0:
        b_train = indexes[split_size:]
        b_test = indexes[:split_size]

    #elif split == int(1 // test_prop) - 1:
    elif split == (size // split_size) - 1:
        b_train = indexes[:split_size * split]
        b_test = indexes[split_size * split:]

    else:
        b_train = np.concatenate([indexes[0:split_size * split], indexes[split_size * (split + 1): ]])
        b_test = indexes[split_size * split : split_size * (split+1)]

    b_val = b_train[:val_size]
    b_train = b_train[val_size:]
    
        
    train_indexes = b_train
    val_indexes = b_val
    test_indexes = b_test
    train_ds = FilteredElementsDs(dataset, train_indexes)
    val_ds = FilteredElementsDs(dataset, val_indexes)
    test_ds = FilteredElementsDs(dataset, test_indexes)
    
    return train_ds, val_ds, test_ds
    
    
class FilteredElementsDs(Dataset):
    def __init__(self, total_dataset, new_indexes):
        self.new_indexes = new_indexes
        self.total_dataset = total_dataset
        
    def __len__(self):
        return len(self.new_indexes)
    
    def __getitem__(self, idx):
        internal_idx = self.new_indexes[idx]
        return self.total_dataset[internal_idx]