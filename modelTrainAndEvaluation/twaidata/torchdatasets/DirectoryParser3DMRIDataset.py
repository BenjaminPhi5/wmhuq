import os
import SimpleITK as sitk
from torch.utils.data import Dataset
from natsort import natsorted
import numpy as np
import torch
from twaidata.mri_dataset_directory_parsers.MSS3_multirater import MSS3MultiRaterDataParser
from twaidata.mri_dataset_directory_parsers.LBC_multirater import LBCMultiRaterDataParser
from twaidata.mri_dataset_directory_parsers.LBC_multirater_kjf import LBCkjfMultiRaterDataParser
from twaidata.mri_dataset_directory_parsers.WMHChallenge_interrater import WMHChallengeInterRaterDirParser

class DirectoryParser3DMRIDataset(Dataset):
    def __init__(self, parser):
        super().__init__()
        
        self.iomap = parser.get_dataset_inout_map()
        self.keys = natsorted(list(self.iomap.keys()))
        
        
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
            
            img = sitk.GetArrayFromImage(sitk.ReadImage(outfile)).astype(np.float32)
            img = torch.from_numpy(img)
            
            if file_key == "FLAIR":
                mask = (img != 0).type(torch.float32)
                xs["mask"] = mask
            
            if islabel:
                ys[file_key] = img
            else:
                xs[file_key] = img
                
        return xs, ys, key
        
    
    def __len__(self):
        return len(self.keys)
    
    
class MSS3InterRaterDataset(DirectoryParser3DMRIDataset):
    def __init__(self):
        super().__init__(
            MSS3MultiRaterDataParser(
        # paths on the cluster for the in house data
        "/home/s2208943/ipdis/data/InterRater_data",
        "/home/s2208943/ipdis/data/preprocessed_data/MSS3_InterRaterData"
            )
        )

class LBCInterRaterDataset(DirectoryParser3DMRIDataset):
    def __init__(self):
        super().__init__(
            LBCMultiRaterDataParser(
        # paths on the cluster for the in house data
        "/home/s2208943/ipdis/data/InterRater_data",
        "/home/s2208943/ipdis/data/preprocessed_data/LBC_InterRaterData"
            )
        )

class LBCkjfInterRaterDataset(DirectoryParser3DMRIDataset):
    def __init__(self):
        super().__init__(
            LBCkjfMultiRaterDataParser(
        # paths on the cluster for the in house data
        "/home/s2208943/ipdis/data/InterRater_data",
        "/home/s2208943/ipdis/data/preprocessed_data/LBCkjf_InterRaterData"
            )
        )
        
class WMHChallengeInterRaterDataset(DirectoryParser3DMRIDataset):
    def __init__(self):
        super().__init__(
            WMHChallengeInterRaterDirParser(
        # paths on the cluster for the in house data
        "/home/s2208943/ipdis/data/WMH_Challenge_full_dataset",
        "/home/s2208943/ipdis/data/preprocessed_data/WMHChallenge_InterRaterData"
            )
        )
