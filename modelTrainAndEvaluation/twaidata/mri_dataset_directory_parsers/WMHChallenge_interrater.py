"""
Directory parser for the WMH challenge dataset
"""

# TODO: sort out link and citation for the dataset in this file
# TODO: sort out proper package imports

from twaidata.mri_dataset_directory_parsers.generic import DirectoryParser
from twaidata.MRI_preprep.io import FORMAT
import os

SUBFOLDER = "pre" # pre is the set of preprocessed data

class WMHChallengeInterRaterDirParser(DirectoryParser):
    """
    structure of WMH dataset:
    
    additional_annotations/
        observer_o3/
            training/
                Amsterdam_GE3T/
                    <id>/
                        result.nii.gz
                    ...
                Singapore/
                    ...
                Utretch/
                    ...
        observer_o4/
            ...
    training/
        Amsterdam_GE3T/
            <id>/
                orig/
                    ... unpreprocessed versions
                pre/
                    3DT1.nii.gz # this one is not aligned with the flair, so I can just ignore it.
                    FLAIR.nii.gz
                    T1.nii.gz
                wmh.nii.gz
        Singapore/
            ...
        Utretch/
            ...
    test/
        ...
    """
    
    def __init__(self, dataset_root_in, *args, **kwargs):
        self.domains = ["Amsterdam_GE3T", "Singapore", "Utrecht"]
        super().__init__(dataset_root_in, *args, **kwargs)
        
    
    def _build_dataset_table(self):
        for domain in self.domains:
            domain_dir = os.path.join(self.root_in, "training", domain)
            individuals = os.listdir(domain_dir)
            
            for ind in individuals:
                # extract T1, FLAIR and mask
                individual_dir = os.path.join(domain_dir, ind)
                t1 = os.path.join(individual_dir, SUBFOLDER, f"T1{FORMAT}")
                flair = os.path.join(individual_dir, SUBFOLDER, f"FLAIR{FORMAT}")
                wmh = os.path.join(individual_dir, f"wmh{FORMAT}")
                wmho3 = os.path.join(self.root_in, "additional_annotations", "observer_o3", "training", domain, ind, "result.nii.gz")
                wmho4 = os.path.join(self.root_in, "additional_annotations", "observer_o4", "training", domain, ind, "result.nii.gz")
                
                ind_files_map = {}
                ind_files_map["T1"] = {
                    "infile":t1,
                    "outpath":os.path.join(*[self.root_out, "imgs"]), 
                    "outfilename":f"{domain}_{ind}_T1",
                    "islabel":False
                }
                
                ind_files_map["FLAIR"] = {
                    "infile":flair,
                    "outpath":os.path.join(*[self.root_out, "imgs"]), 
                    "outfilename":f"{domain}_{ind}_FLAIR",
                    "islabel":False
                }
                
                ind_files_map["wmh"] = {
                    "infile":wmh,
                    "outpath":os.path.join(*[self.root_out, "labels"]), 
                    "outfilename":f"{domain}_{ind}_wmh",
                    "islabel":True
                }
                
                ind_files_map["wmho3"] = {
                    "infile":wmho3,
                    "outpath":os.path.join(*[self.root_out, "labels"]), 
                    "outfilename":f"{domain}_{ind}_wmho3",
                    "islabel":True
                }
                
                ind_files_map["wmho4"] = {
                    "infile":wmho4,
                    "outpath":os.path.join(*[self.root_out, "labels"]), 
                    "outfilename":f"{domain}_{ind}_wmho4",
                    "islabel":True
                }
                
                self.files_map[f"{domain}_{ind}"] = ind_files_map
    
    
if __name__ == "__main__":
    print("testing")
    parser = WMHChallengeInterRaterDirParser(
        # paths on the cluster for the in house data
        "/home/s2208943/ipdis/data/WMH_Challenge_full_dataset",
        "/home/s2208943/ipdis/data/preprocessed_data/WMHChallenge_InterRaterData"
    )
    
    iomap = parser.get_dataset_inout_map()
    for key, value in iomap.items():
        print("individual: ", key)
        print("individual map:", value)