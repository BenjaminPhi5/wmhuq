"""
Directory parser for the WMH challenge dataset

I want to now build a unified challenge dataset. I should split the dataset into the domains explicitly like I did for the CVD dataset, since that is what that code expects
Then I also need to split it into training and testing. I can do this flat, so training_domain1, testing_domain1 etc.
"""

# TODO: sort out link and citation for the dataset in this file
# TODO: sort out proper package imports

from twaidata.mri_dataset_directory_parsers.generic import DirectoryParser
from twaidata.MRI_preprep.io import FORMAT
import os

SUBFOLDER = "orig" # dataset also has a pre folder which I am ignoring for now

class WMHChallengeDirParser(DirectoryParser):
    """
    structure of WMH dataset:
    
    /public
        GE3T/
            <id>/
                orig/
                    T1.nii.gz
                    FLAIR.nii.gz
                    ...         (other files, some regression data and '3D' versions)
                pre/
                    ...? whats in here?
                wmh.nii.gz
        Singapore/
            ...
        Utretch/
            ...
    """
    
    def __init__(self, dataset_root_in, *args, **kwargs):
        self.domains = ["GE3T", "Singapore", "Utrecht"]
         # move into the /public folder
        dataset_root_in = os.path.join(dataset_root_in, "public")
        
        super().__init__(dataset_root_in, *args, **kwargs)
        
    
    def _build_dataset_table(self):
        for domain in self.domains:
            domain_dir = os.path.join(self.root_in, domain)
            individuals = os.listdir(domain_dir)
            
            for ind in individuals:
                # extract T1, FLAIR and mask
                individual_dir = os.path.join(domain_dir, ind)
                t1 = os.path.join(*[individual_dir, SUBFOLDER, f"T1{FORMAT}"])
                flair = os.path.join(*[individual_dir, SUBFOLDER, f"FLAIR{FORMAT}"])
                mask = os.path.join(individual_dir, f"wmh{FORMAT}")
                
                
                # add to map the T1, FLAIR, and mask
                # note filetypes will be automatically added later in preprocessing
                
                ind_files_map = {}
                ind_files_map["T1"] = {
                    "infile":t1,
                    "outpath":os.path.join(*[self.root_out, domain, "imgs"]), 
                    "outfilename":f"{ind}_T1",
                    "islabel":False
                }
                
                ind_files_map["FLAIR"] = {
                    "infile":flair,
                    "outpath":os.path.join(*[self.root_out, domain, "imgs"]), 
                    "outfilename":f"{ind}_FLAIR",
                    "islabel":False
                }
                
                ind_files_map["wmh"] = {
                    "infile":mask,
                    "outpath":os.path.join(*[self.root_out, domain, "labels"]), 
                    "outfilename":f"{ind}_wmh",
                    "islabel":True
                }
                
                self.files_map[ind] = ind_files_map
    
    
if __name__ == "__main__":
    print("testing")
    parser = WMHChallengeDirParser(
        # "/media/benp/NVMEspare/datasets/MRI_IP_project/WMH_challenge_dataset/",
        # "/media/benp/NVMEspare/datasets/preprocessing_attempts/local_results"
        # replace with the directory to the dataset on your drive
        "/home/s2208943/ipdis/data/extra_data/MRI_IP_project/WMH_challenge_dataset/",
        "/home/s2208943/ipdis/data/preprocessed_data/WMH_challenge_dataset"
    )
    
    iomap = parser.get_dataset_inout_map()
    for key, value in iomap.items():
        print("individual: ", key)
        print("individual map:", value)