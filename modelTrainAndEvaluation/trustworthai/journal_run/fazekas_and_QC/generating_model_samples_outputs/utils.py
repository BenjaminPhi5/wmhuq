from torch.utils.data import Dataset
import os
import numpy as np
from tqdm import tqdm
import torch

class LimitedSizeDataset(Dataset):
    def __init__(self, base_dataset, start, end):
        super().__init__()
        self.base_dataset = base_dataset
        self.start = start
        self.end = end
        
    def __len__(self):
        return self.end - self.start
    
    def __getitem__(self, idx):
        if idx + self.start >= self.end:
            raise IndexError
        return self.base_dataset[idx + self.start]
    
def write_results_to_disk(args, IDs, samples_all, output_dir="/home/s2208943/preprocessed_data/ADNI_300_output_maps"):
    # create the output folder
    # output_dir = "/home/s2208943/ipdis/data/preprocessed_data/ADNI_300_output_maps"
    uncertainty_type = args.uncertainty_type
    
    # write each file of combined outputs to disk. Nice.
    for i in tqdm(range(len(IDs)), position=0, leave=True):
        ID = IDs[i]
        samples = samples_all[i]
        samples = torch.nn.functional.softmax(samples.cuda(), dim=2)[:,:,1]
        np.savez_compressed(os.path.join(output_dir, uncertainty_type, f'{ID}_model_WMH_samples.npz'), samples=samples.cpu())
