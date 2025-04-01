"""
Directory parser for the WMH challenge dataset, containing the training data, the testing data, and the additional annotations.

I want to now build a unified challenge dataset. I should split the dataset into the domains explicitly like I did for the CVD dataset, since that is what that code expects
Then I also need to split it into training and testing. I can do this flat, so training_domain1, testing_domain1 etc.
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
        self.domains = ['Amsterdam_GE3T', 'Singapore', 'Utrecht', 'Amsterdam_GE1T5', 'Amsterdam_Philips_VU_PETMR_01']
        self.folds = ['training', 'test']
        super().__init__(dataset_root_in, *args, **kwargs)
        
    
    def _build_dataset_table(self):
        for fold in self.folds:
            for domain in self.domains:
                domain_dir = os.path.join(self.root_in, fold, domain)
                if not os.path.exists(domain_dir):
                    continue # some are held out for testing only
                    
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
                        "outpath":os.path.join(*[self.root_out, fold, domain, "imgs"]), 
                        "outfilename":f"{ind}_T1",
                        "islabel":False,
                    }

                    ind_files_map["FLAIR"] = {
                        "infile":flair,
                        "outpath":os.path.join(*[self.root_out, fold, domain, "imgs"]), 
                        "outfilename":f"{ind}_FLAIR",
                        "islabel":False,
                    }

                    ind_files_map["wmh"] = {
                        "infile":wmh,
                        "outpath":os.path.join(*[self.root_out, fold, domain, "labels"]), 
                        "outfilename":f"{ind}_wmh",
                        "islabel":True,
                    }

                    if fold == "training":
                        ind_files_map["wmho3"] = {
                            "infile":wmho3,
                            "outpath":os.path.join(*[self.root_out, fold, domain, "labels"]), 
                            "outfilename":f"{ind}_wmho3",
                            "islabel":True,
                        }

                        ind_files_map["wmho4"] = {
                            "infile":wmho4,
                            "outpath":os.path.join(*[self.root_out, fold, domain, "labels"]), 
                            "outfilename":f"{ind}_wmho4",
                            "islabel":True,
                        }

                    self.domain_map[f"{fold}_{domain}_{ind}"] = f"{fold}_{domain}"
                    self.files_map[f"{fold}_{domain}_{ind}"] = ind_files_map

    
if __name__ == "__main__":
    print("testing")
    parser = WMHChallengeInterRaterDirParser(
        # paths on the cluster for the in house data
        "/home/s2208943/datasets/WMH_Challenge",
        "/home/s2208943/preprocessed_data/WMHChallenge_InterRaterData"
    )
    
    iomap = parser.get_dataset_inout_map()
    for key, value in iomap.items():
        print("individual: ", key)
        print("individual map:", value)