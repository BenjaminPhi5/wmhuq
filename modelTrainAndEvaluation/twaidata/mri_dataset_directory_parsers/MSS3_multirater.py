"""
directory parser for the MSS3 multi rater dataset
"""

from twaidata.mri_dataset_directory_parsers.generic import DirectoryParser
from twaidata.MRI_preprep.io import FORMAT
import os
import importlib.resources as pkg_resources


class MSS3MultiRaterDataParser(DirectoryParser):
    """
    structure of MSS3MultiRaterDataParser:
    MSS3_ED_001/
        V1/
            MSS3_ED_001_V1_FLAIRbrain.nii.gz
            MSS3_ED_001_V1_lacune.nii.gz
            MSS3_ED_001_V1_T1Wbrain.nii.gz
            MSS3_ED_001_V1_T2Wbrain.nii.gz
            MSS3_ED_001_V1_WMH_mask_ES.nii.gz
            MSS3_ED_001_V1_WMH_mask_MVH.nii.gz
        V2/
            ...
    ...
    ...
    MSS3_ED_079/
        ...
    """ 
    def _build_dataset_table(self):
        folders = os.listdir(self.root_in)
        
        for ind in folders:
            if "MSS3" not in ind:
                continue
            
            subfolders = os.listdir(os.path.join(self.root_in, ind))
            for vf in subfolders:
                if "V" not in vf:
                    continue
                
                files = os.listdir(os.path.join(self.root_in, ind, vf))
                ind_files_map = {}
                for f in files:
                    fpath = os.path.join(self.root_in, ind, vf, f)
                    if "T1Wbrain" in f and "seg" not in f:
                        ind_files_map["T1"] = {
                            "infile":fpath,
                            "outpath":os.path.join(self.root_out, "imgs"), 
                            "outfilename":f"{ind}_{vf}_T1",
                            "islabel":False
                        }
                    elif "FLAIRbrain" in f:
                        ind_files_map["FLAIR"] = {
                            "infile":fpath,
                            "outpath":os.path.join(self.root_out, "imgs"), 
                            "outfilename":f"{ind}_{vf}_FLAIR",
                            "islabel":False
                        }
                    elif "WMH_mask_ES" in f:
                        ind_files_map["wmhes"] = {
                            "infile":fpath,
                            "outpath":os.path.join(self.root_out, "labels"), 
                            "outfilename":f"{ind}_{vf}_wmhes",
                            "islabel":True
                        }
                    elif "WMH_mask_MVH" in f:
                        ind_files_map["wmhmvh"] = {
                            "infile":fpath,
                            "outpath":os.path.join(self.root_out, "labels"), 
                            "outfilename":f"{ind}_{vf}_wmhmvh",
                            "islabel":True
                        }
                    elif "lacune" in f:
                        ind_files_map["lacune"] = {
                            "infile":fpath,
                            "outpath":os.path.join(self.root_out, "labels"), 
                            "outfilename":f"{ind}_{vf}_lacune",
                            "islabel":True
                        }
                        
                    self.files_map[f"{ind}_{vf}"] = ind_files_map
    
    
if __name__ == "__main__":
    print("testing")
    parser = MSS3MultiRaterDataParser(
        # paths on the cluster for the in house data
        "/home/s2208943/ipdis/data/InterRater_data",
        "/home/s2208943/ipdis/data/preprocessed_data/MSS3_InterRaterData"
    )
    
    iomap = parser.get_dataset_inout_map()
    for key, value in iomap.items():
        print("individual: ", key)
        print("individual map:", value)