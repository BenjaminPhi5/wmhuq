from torch.utils.data import Dataset
import pandas as pd
from twaidata.torchdatasets.whole_brain_dataset import MRISegmentationDatasetFromFile
import os

def get_clinscores(imgs_dir, clinScores_filepath, domain):
    # get list of individuals (IDs) for that file
    files_ds = MRISegmentationDatasetFromFile(os.path.join(imgs_dir, domain), 
                 img_filetypes=["FLAIR_BET_mask.nii.gz", "FLAIR.nii.gz", "T1.nii.gz"], # brain mask, flair, T1.
                 label_filetype="wmh.nii.gz")
    individuals = files_ds.individuals
    
    df = pd.read_csv(clinScores_filepath)
    filtered = df.loc[df['ID'].isin(individuals)]

    return filtered


class ImgAndClinScoreDataset3d(Dataset):
    def __init__(self, in_disk_img_dir, clin_path, base_dataset, domain, transforms=None):
        super().__init__()
        self.base_dataset = base_dataset
        self.clinscore_df = get_clinscores(in_disk_img_dir, clin_path, domain)
        assert len(self.base_dataset) == len(self.clinscore_df)
        self.transforms = transforms
        
    def __getitem__(self, idx):
        pair = self.base_dataset[idx]
        
        if self.transforms:
            pair = self.transforms(pair)
            
        return (*pair, self.clinscore_df.iloc[idx])
    
    def __len__(self):
        return len(self.base_dataset)