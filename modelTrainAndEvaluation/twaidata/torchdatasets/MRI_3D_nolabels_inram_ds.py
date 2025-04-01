import os
import SimpleITK as sitk
from torch.utils.data import Dataset
from collections import defaultdict
from natsort import natsorted
import numpy as np
import torch
from collections import defaultdict
from tqdm import tqdm

class MRI_3D_nolabels_inram_ds(Dataset):
    def __init__(self, imgs_folder, returned_filetypes=["FLAIR.nii.gz", "FLAIR_BET_mask.nii.gz", "T1.nii.gz"]):
        super().__init__()
        """
        this is a more general dataset that just assumes there is a folder
        containing images named e.g ID_["FLAIR.nii.gz", "FLAIR_BET_mask.nii.gz", "T1.nii.gz"]
        it returns the id for each example.
        """
        # find all the relevant file names and group them by ID
        ind_files = defaultdict(lambda : {})
        files = os.listdir(imgs_folder)
        for file in files:
            for filetype in returned_filetypes:
                if filetype in file:
                    # extracting the ID from the filename: assumes all files are ID_<filetype>
                    ID = file[0:-len(filetype)-1]
                    #print(file, ID)
                    ind_files[ID][filetype] = file
                    
        # check that each individual has all filetypes
        for ID, files_map in ind_files.items():
            if len(files_map.keys()) != len(returned_filetypes):
                raise ValueError(f"not all files found for ID: {ID}")
                    
        # load all of the images into memory and stack them together
        images_lists = []
        IDs = []
        for ID in tqdm(natsorted(list(ind_files.keys())), position=0, leave=True):
            ID_images = []
            for i, filetype in enumerate(returned_filetypes):
                ID_images.append(
                    sitk.GetArrayFromImage(
                        sitk.ReadImage(os.path.join(imgs_folder, ind_files[ID][filetype]))
                    ).astype(np.float32)
                )
            images_lists.append(ID_images)
            IDs.append(ID)
        
        images_lists = [
            torch.from_numpy(np.stack(id_imgs)) for id_imgs in images_lists
        ]
        self.images_lists = images_lists
        self.IDs = IDs
        
    def __getitem__(self, idx):
        return self.images_lists[idx]
    
    def getIDs(self):
        return self.IDs
    
    def __len__(self):
        return len(self.images_lists)
