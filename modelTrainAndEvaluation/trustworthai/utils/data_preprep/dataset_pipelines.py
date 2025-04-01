"""
TODO: I am going to need to

[x] pipeline for our 2D training dataset
[x] 3d evaluation dataset
[x] 2d and 3d for the wmh challenge dataset as well
[ ] clinscores dataset
[x] allow setting hyperparameters (remember the filter empties dataset)

"""

from trustworthai.utils.fitting_and_inference.get_scratch_dir import scratch_dir
import os
from torch.utils.data import ConcatDataset, DataLoader
from twaidata.torchdatasets_v2.mri_dataset_inram import MRISegmentation3DDataset, CSVDataWrapper
from trustworthai.utils.data_preprep.splits import train_val_test_split
from trustworthai.utils.data_preprep.splits import cross_validate_split
from trustworthai.utils.data_preprep.splits import FilteredElementsDs
from trustworthai.utils.data_preprep.brain_to_slices_dataset import MRISegDataset2DFrom3D
from trustworthai.utils.data_preprep.empty_slice_filtering import FilteredEmptyElementsDataset
from trustworthai.utils.augmentation.augmentation_pipelines import get_transforms
from twaidata.torchdatasets.clinScores_dataset import get_clinscores, ImgAndClinScoreDataset3d
import numpy as np

# root_dir = os.path.join(scratch_dir(), "preprep/out_data/collated/")
root_dir = "/home/s2208943/preprocessed_data/"
wmh_dir = root_dir + "WMHChallenge_InterRaterData/collated/"
ed_dir = root_dir + "Ed_CVD/collated/"

domains_ed = [
            ed_dir + d for d in ["domainA", "domainB", "domainC", "domainD"]
          ]

domains_chal = [
    wmh_dir + d for d in ["training_Singapore", "training_Utrecht", "training_Amsterdam_GE3T"]
]


def load_data(dataset="ed", test_proportion=0.15, validation_proportion=0.15, seed=3407, empty_proportion_retained=0.1, batch_size=32, dataloader2d_only=True, dataset3d_only=False,
             cross_validate=False, cv_split=None, cv_test_fold_smooth=1, merge_val_test=False):
    if dataset == "ed":
        domains = domains_ed
        data_dir = ed_dir
        label_key = None
        label_value = None
    elif dataset == "chal":
        domains = domains_chal
        data_dir = wmh_dir
        label_key = "wmh"
        label_value = 1
    else:
        raise ValueError(f"dataset {dataset} not defined, only ed or chal accepted")
    
    # 1 - load the 3D images as a dataset
    datasets_domains = [
        MRISegmentation3DDataset(
            data_dir, domain_name=domain, transforms=None, xy_only=True, label_key=label_key, label_value=label_value,
        ) 
        for domain in domains
    ]

    # 2 - split into train, val test datasets per domain
    train_dataset_3d, val_dataset_3d, test_dataset_3d = domains_to_splits(datasets_domains, validation_proportion, test_proportion, seed, cross_validate, cv_split, cv_test_fold_smooth=cv_test_fold_smooth)
    
    # 3 return 3d if only 3d required
    if dataset3d_only:
        return train_dataset_3d, val_dataset_3d, test_dataset_3d
    
    print(len(train_dataset_3d), len(val_dataset_3d), len(test_dataset_3d))

    # 4 - convert the 3d images to a 2d axial slice dataset
    datasets_2d = [MRISegDataset2DFrom3D(ds, transforms=None) for ds in [train_dataset_3d, val_dataset_3d, test_dataset_3d]]

    # 5 - remove a proportion of axial slices with no label
    train_dataset, val_dataset, test_dataset = [FilteredEmptyElementsDataset(ds, seed=seed, transforms=get_transforms(), empty_proportion_retained=empty_proportion_retained) for ds in datasets_2d]
    
    if merge_val_test.lower() == 'true':
            val_dataset = ConcatDataset([val_dataset, test_dataset])

    # 6 - create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    if merge_val_test.lower() != 'true':
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    else:
        test_dataloader = None
        test_dataset = None
    
    if not dataloader2d_only:
        return {
            "train_dataset3d":train_dataset_3d,
            "val_dataset3d":val_dataset_3d,
            "test_dataset3d":test_dataset_3d,

            "train_dataset2d":train_dataset,
            "val_dataset2d":val_dataset,
            "test_dataset2d":test_dataset,

            "train_dataloader2d":train_dataloader,
            "val_dataloader2d":val_dataloader,
            "test_dataloader2d":test_dataloader,
        }
    
    return train_dataloader, val_dataloader, test_dataloader
    

class StandardDataRetriever():
    def __init__(self, dataset):
        if dataset == "ed":
            domains = domains_ed
            label_key = None
            label_value = None
        elif dataset == "chal":
            domains = domains_chal
            label_key = "wmh"
            label_value = 1
        else:
            raise ValueError(f"dataset {dataset} not defined, only ed or chal accepted")

        # 1 - load the 3D images as a dataset
        self.datasets_domains = [MRISegmentation3DDataset(root_dir, domain, transforms=None) for domain in domains]
    
    def load_data(self, test_proportion=0.15, validation_proportion=0.15, seed=3407, empty_proportion_retained=0.1, batch_size=32, dataloader2d_only=True, dataset3d_only=False,
             cross_validate=False, cv_split=None, cv_test_fold_smooth=1, merge_val_test=False):
        
        # 2 - split into train, val test datasets per domain
        train_dataset_3d, val_dataset_3d, test_dataset_3d = domains_to_splits(self.datasets_domains, validation_proportion, test_proportion, seed, cross_validate, cv_split, cv_test_fold_smooth=cv_test_fold_smooth)

        # 3 return 3d if only 3d required
        if dataset3d_only:
            return train_dataset_3d, val_dataset_3d, test_dataset_3d

        # 4 - convert the 3d images to a 2d axial slice dataset
        datasets_2d = [MRISegDataset2DFrom3D(ds, transforms=None) for ds in [train_dataset_3d, val_dataset_3d, test_dataset_3d]]

        # 5 - remove a proportion of axial slices with no label
        train_dataset, val_dataset, test_dataset = [FilteredEmptyElementsDataset(ds, seed=seed, transforms=get_transforms(), empty_proportion_retained=empty_proportion_retained) for ds in datasets_2d]
        
        if merge_val_test.lower() == 'true':
            val_dataset = ConcatDataset([val_dataset, test_dataset])

        # 6 - create dataloaders
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        
        if merge_val_test.lower() != 'true':
            test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        else:
            test_dataloader = None
            test_dataset = None

        if not dataloader2d_only:
            return {
                "train_dataset3d":train_dataset_3d,
                "val_dataset3d":val_dataset_3d,
                "test_dataset3d":test_dataset_3d,

                "train_datset2d":train_dataset,
                "val_dataset2d":val_dataset,
                "test_dataset2d":test_dataset,

                "train_dataloader2d":train_dataloader,
                "val_dataloader2d":val_dataloader,
                "test_dataloader2d":test_dataloader,
            }

        return train_dataloader, val_dataloader, test_dataloader
    
def load_clinscores_data():
    pass
    
class ClinScoreDataRetriever():
    def __init__(self, use_updated_scores=True):
        
        if use_updated_scores == False:
            print("The use_updated_scores now defaults to True and the flag is ignored")
            
        clin_path = "/home/s2208943/preprocessed_data/Ed_CVD/cinscore_data.csv"

        domains = domains_ed
        data_dir = ed_dir
        label_key = None
        label_value = None
    
        datasets_domains = [
            MRISegmentation3DDataset(
                data_dir, domain_name=domain, transforms=None, xy_only=False, label_key=label_key, label_value=label_value,
            ) 
            for domain in domains
        ]

        # old code from before data rearrangement commented out
        # domain_names = ["domainA", "domainB", "domainC", "domainD"]
        # self.clinscore_datasets = [ImgAndClinScoreDataset3d(in_disk_img_dir, clin_path, datasets_domains[i], domain_names[i], transforms=None) for i in range(len(domains))]
        self.clinscore_datasets = [CSVDataWrapper(domain_ds, "/home/s2208943/preprocessed_data/ADNI300/clinscore_data.csv") for domain_ds in datasets_domains]

    def load_clinscores_data(self, combine_all=True, validation_proportion=0.15, test_proportion=0.15, seed=3407, cross_validate=False, cv_split=None, cv_test_fold_smooth=1):
        train_clinscore_ds, val_clinscore_ds, test_clinscore_ds = domains_to_splits(self.clinscore_datasets, validation_proportion=validation_proportion, test_proportion=test_proportion, seed=seed, cross_validate=cross_validate, cv_split=cv_split, cv_test_fold_smooth=cv_test_fold_smooth)
    
        if combine_all:
            combined_ds = ConcatDataset([train_clinscore_ds, val_clinscore_ds, test_clinscore_ds])
            return combined_ds

        else:
            return train_clinscore_ds, val_clinscore_ds, test_clinscore_ds
        

def domains_to_splits(domain_datasets, validation_proportion, test_proportion, seed, cross_validate, cv_split, cv_test_fold_smooth=1):
    
    if not cross_validate:
        datasets_3d = [train_val_test_split(dataset, validation_proportion, test_proportion, seed) for dataset in domain_datasets]
    else:
        datasets_3d = [cross_validate_split(dataset, validation_proportion, test_proportion, seed, cv_split, test_fold_smooth=cv_test_fold_smooth) for dataset in domain_datasets]

    #datasets_3d = [train_val_test_split(dataset, validation_proportion, test_proportion, seed) for dataset in domain_datasets]

    # concat the train val test datsets
    train_dataset_3d = ConcatDataset([ds[0] for ds in datasets_3d])
    val_dataset_3d = ConcatDataset([ds[1] for ds in datasets_3d])
    test_dataset_3d = ConcatDataset([ds[2] for ds in datasets_3d])
    return train_dataset_3d, val_dataset_3d, test_dataset_3d