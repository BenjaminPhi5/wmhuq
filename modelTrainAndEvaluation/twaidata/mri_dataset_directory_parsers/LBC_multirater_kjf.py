"""
directory parser for the LBC multi rater dataset that contains the kjf segmentations
"""

from twaidata.mri_dataset_directory_parsers.generic import DirectoryParser
from twaidata.MRI_preprep.io import FORMAT
import os
import importlib.resources as pkg_resources
from natsort import natsorted


class LBCkjfMultiRaterDataParser(DirectoryParser):
    """
    structure of LBCMultiRaterDataParser:
    
    LBC1921_20930/
        LBC1921_20930_brainmask.nii.gz
        LBC1921_20930_FLAIRbrain.nii.gz
        LBC1921_20930_ICV.nii.gz
        LBC1921_20930_T1Wbrain.nii.gz
        LBC1921_20930_WMH2.nii.gz
        LBC1921_20930_WMH5.nii.gz
        LBC1921_20930_CSF.nii.gz
        LBC1921_20930_GM.nii.gz
        LBC1921_20930_NAWM.nii.gz
        LBC1921_20930_T2Wbrain.nii.gz
        LBC1921_20930_WMH4.nii.gz
    ...
    ...
    LBC1921_20947/
        ...
    """
    
    def __init__(self, dataset_root_in, *args, **kwargs):
        super().__init__(dataset_root_in, *args, **kwargs)
        
    
    def _build_dataset_table(self):
        folders = os.listdir(self.root_in)
        
        ### todo to deal with:
        # if the folder has 1 an 4 versions in it then take the 4 version.
        # some of the files are .nii and so the easiest thing to do is convert with replacement?
        # for both datasets, I need to check whether the images are
        # sampled to the same space or not, looking that the wmh segmentation maps. I will need to load a few slices to see for myself. Good.
        # then we will likely need a custom dataset to load these brains.. hmmm that may be a bit of a pain actually. Hurrumph.
        
        for ind in folders:
            if "LBC" not in ind:
                continue
                
            files = natsorted(os.listdir(os.path.join(self.root_in, ind)))
            has_kjf_version = False
            for f in files:
                if "wmh" in f.lower() and "kjf" in f.lower():
                    has_kjf_version = True
            
            if not has_kjf_version: # these are the only images that seem to have genuine inter-rater variability. However they use a different method for arriving at the inter-rater information.
                continue
            
            has_v4 = False
            for f in files:
                if "_4_" in f:
                    has_v4 = True
                    break
            ind_files_map = {}
            for f in files:
                if has_v4 and "_4_" not in f:
                    continue
                if ".nii.gz" not in f:
                    continue
                
                fpath = os.path.join(self.root_in, ind, f)
                if "t1" in f.lower() and "brain" in f.lower() and "seg" not in f.lower():
                    ind_files_map["T1"] = {
                        "infile":fpath,
                        "outpath":os.path.join(self.root_out, "imgs"), 
                        "outfilename":f"{ind}_T1",
                        "islabel":False
                    }
                elif "flair" in f.lower() and "brain" in f.lower():
                    ind_files_map["FLAIR"] = {
                        "infile":fpath,
                        "outpath":os.path.join(self.root_out, "imgs"), 
                        "outfilename":f"{ind}_FLAIR",
                        "islabel":False
                    }
                elif "wmh" in f.lower() and "kjf" in f.lower():
                    ind_files_map[f"wmh_kjf"] = {
                        "infile":fpath,
                        "outpath":os.path.join(self.root_out, "labels"), 
                        "outfilename":f"{ind}_wmh_kjf",
                        "islabel":True
                    }                                  
                elif "wmh" in f.lower() and "thresholding" not in f.lower():
                    wmh_id = f.lower().split("wmh")[1].split(".")[0]
                    ind_files_map[f"wmh{wmh_id}"] = {
                        "infile":fpath,
                        "outpath":os.path.join(self.root_out, "labels"), 
                        "outfilename":f"{ind}_wmh{wmh_id}",
                        "islabel":True
                    }
                elif "icv" in f.lower():
                    continue # the ICVs are causing problems and need to be ignored.
                    # ind_files_map["ICV"] = {
                    #     "infile":fpath,
                    #     "outpath":None, 
                    #     "outfilename":None,
                    #     "islabel":False
                    # }

                self.files_map[ind] = ind_files_map
    
    
if __name__ == "__main__":
    print("testing")
    parser = LBCkjfMultiRaterDataParser(
        # paths on the cluster for the in house data
        "/home/s2208943/ipdis/data/InterRater_data",
        "/home/s2208943/ipdis/data/preprocessed_data/LBCkjf_InterRaterData"
    )
    
    iomap = parser.get_dataset_inout_map()
    for key, value in iomap.items():
        print("individual: ", key)
        print("individual map:", value)