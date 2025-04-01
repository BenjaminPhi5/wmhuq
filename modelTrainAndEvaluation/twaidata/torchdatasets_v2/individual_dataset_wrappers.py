from twaidata.torchdatasets_v2.mri_dataset_from_file import MRISegmentationDatasetFromFile
from twaidata.mri_dataset_directory_parsers.MSS3_multirater import MSS3MultiRaterDataParser
from twaidata.mri_dataset_directory_parsers.LBC_multirater import LBCMultiRaterDataParser
from twaidata.mri_dataset_directory_parsers.LBC_multirater_kjf import LBCkjfMultiRaterDataParser
from twaidata.mri_dataset_directory_parsers.WMHChallenge_Unified import WMHChallengeInterRaterDirParser

class MSS3InterRaterDataset(MRISegmentationDatasetFromFile):
    def __init__(self):
        super().__init__(
            MSS3MultiRaterDataParser(
        # paths on the cluster for the in house data
        "/home/s2208943/datasets/Inter_observer",
        "/home/s2208943/preprocessed_data/MSS3_InterRaterData"
    ),
            extra_filetypes=["mask", "vent_distance"]
        )

class LBCInterRaterDataset(MRISegmentationDatasetFromFile):
    def __init__(self):
        super().__init__(
            LBCMultiRaterDataParser(
#         # paths on the cluster for the in house data
        "/home/s2208943/datasets/Inter_observer",
        "/home/s2208943/preprocessed_data/LBC_InterRaterData"
    ),
            extra_filetypes=["mask", "vent_distance"]
        )
        
class WMHChallengeInterRaterDataset(MRISegmentationDatasetFromFile):
    def __init__(self):
        super().__init__(
            WMHChallengeInterRaterDirParser(
    "/home/s2208943/datasets/WMH_Challenge",
    "/home/s2208943/preprocessed_data/WMHChallenge_InterRaterData"
),
            extra_filetypes=["mask", "vent_distance"]
        )
