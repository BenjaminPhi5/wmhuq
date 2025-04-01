"""
directory parser for the Ed in house dataset
"""

from twaidata.mri_dataset_directory_parsers.generic import DirectoryParser
from twaidata.MRI_preprep.io import FORMAT
import os
import importlib.resources as pkg_resources
import twaidata.mri_dataset_directory_parsers.ed_domains_map as edm

# TODO: sort out link and citation for the dataset in this file
# TODO: sort out proper package imports

class EdDataParser(DirectoryParser):
    """
    structure of ED dataset:
    
    CVD001/
        masks/
            ICV.nii.gz
            WMH.nii.gz
            ... (others such as WMH_oud_cortical_stroke.nii.gz) - interesting
        MRI/
            FLAIR.nii.gz
            GRE.nii.gz
            T1.nii.gz
            T2.nii.gz
    ...
    ...
    CVD325/
        ...
    """
    
    def __init__(self, dataset_root_in, *args, **kwargs):
        self.domains = ["domainA", "domainB", "domainC", "domainD"]
        super().__init__(dataset_root_in, *args, **kwargs)
        
    
    def _build_dataset_table(self):
        # load map of domains
        domain_files = {
            # load each domain map file
            d: pkg_resources.read_text(edm, d + ".txt")
            for d in self.domains
        }
        domain_files = {
            # split into separate lines, drop the last line.
            d:text.split("\n")[:-1]
            for (d, text) in domain_files.items()
        }
        # for each item, extract the id of the image
        domain_ids = {
            d: [f.split("_")[0] for f in filelist]
            for (d, filelist) in domain_files.items()
        }
        
        # switch the mapping from id to domain
        ids_to_domain = {
            ind:d
            for (d, inds) in domain_ids.items()
            for ind in inds
        }
        
        individuals = os.listdir(self.root_in)
        
        for ind in individuals:
            if "CVD" not in ind:
                continue # skip the txt readme file
                
            # get domain of individual
            ind_id = ind.split("CVD")[1] # remove the CVD part from the folder name to get the id
            ind_domain = ids_to_domain[ind_id]
            
            # get source filepaths
            t1 = os.path.join(*[self.root_in, ind, "MRI", f"T1{FORMAT}"])
            flair = os.path.join(*[self.root_in, ind, "MRI", f"FLAIR{FORMAT}"])
            mask = os.path.join(*[self.root_in, ind, "masks", f"WMH{FORMAT}"])
            icv = os.path.join(*[self.root_in, ind, "masks", f"ICV{FORMAT}"])


            # add to map the T1, FLAIR, and mask
            # note filetypes will be automatically added later in preprocessing
            ind_files_map = {}
            ind_files_map["T1"] = {
                "infile":t1,
                "outpath":os.path.join(*[self.root_out, ind_domain, "imgs"]), 
                "outfilename":f"{ind}_T1",
                "islabel":False,
            }

            ind_files_map["FLAIR"] = {
                "infile":flair,
                "outpath":os.path.join(*[self.root_out, ind_domain, "imgs"]), 
                "outfilename":f"{ind}_FLAIR",
                "islabel":False,
            }

            ind_files_map["wmh"] = {
                "infile":mask,
                "outpath":os.path.join(*[self.root_out, ind_domain, "labels"]), 
                "outfilename":f"{ind}_wmh",
                "islabel":True,
            }
            
            ind_files_map["ICV"] = { # intracranial volume, just used for preprocessing instead of uing BET!
                "infile":icv,
                "outpath":os.path.join(*[self.root_out, ind_domain, "imgs"]),
                "outfilename":f"{ind}_mask",
                "islabel":True,           # the ICV is a binary map but it is stored in the images folder as it is not a 
                                          # target label for training. I should refactor islabel to is_discrete
            }
            
            self.domain_map[ind] = ind_domain
            self.files_map[ind] = ind_files_map
    
    
if __name__ == "__main__":
    print("testing")
    parser = EdDataParser(
        # paths on the cluster for the in house data
        "/home/s2208943/datasets/CVD/mixedCVDrelease",
        "/home/s2208943/preprocessed_data/Ed_CVD_data"
    )
    
    iomap = parser.get_dataset_inout_map()
    for key, value in iomap.items():
        print("individual: ", key)
        print("individual map:", value)