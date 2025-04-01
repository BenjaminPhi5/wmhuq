from twaibrain.braintorch.fitting_and_inference.get_scratch_dir import scratch_dir
import os
from torch.utils.data import ConcatDataset, DataLoader
from twaibrain.braintorch.data.legacy_dataset_types import *
from twaibrain.brainexperiments.run_nnUNet_v2.old_dataloading.splits import *
from twaibrain.braintorch.augmentation.nnunet_augmentations import get_nnunet_transforms, get_val_transforms, get_simple_transforms
from twaibrain.braintorch.data.legacy_dataset_types.dataset_wrappers import MonaiAugmentedDataset
from twaibrain.braintorch.augmentation.augmentation_pipelines import get_transforms
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

domains_chal_full = [
    wmh_dir + d for d in ["training_Singapore", "test_Singapore", "training_Utrecht", "test_Utrecht", "training_Amsterdam_GE3T", "test_Amsterdam_GE3T", "test_Amsterdam_Philips_VU_PETMR_01", "test_Amsterdam_GE1T5"]
]



def load_data(dataset="ed", test_proportion=0.15, validation_proportion=0.15, seed=3407, empty_proportion_retained=0.1, batch_size=32, dataloader2d_only=True, dataset3d_only=False, dataloader3d_only=False,
             cross_validate=True, cv_split=None, cv_test_fold_smooth=1, merge_val_test=False, num_workers=4, nnUnet_augment=True):
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
    elif dataset == "chalfull":
        domains = domains_chal_full
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

    dims = 3 if dataloader3d_only else 2
    out_spatial_dims = (48, 192, 192) if dims == 3 else (256, 256)
    nntransforms = get_nnunet_transforms(
        axial_only=False,
        dims=dims,
        out_spatial_dims=out_spatial_dims,
        allow_invert=True,
        one_hot_encode=False,
        target_class=1
    )

    simple_transforms = get_simple_transforms(
        axial_only=False,
        dims=dims,
        out_spatial_dims=out_spatial_dims,
        allow_invert=True,
        one_hot_encode=False,
        target_class=1
    )
    
    noaugtransforms = get_val_transforms(
        dims=dims,
        out_spatial_dims=out_spatial_dims,
        one_hot_encode=False, 
        target_class=1
    )

    train_transforms = nntransforms if nnUnet_augment else simple_transforms
    
    if dataloader3d_only:
        train_dataset_3d = MonaiAugmentedDataset(train_dataset_3d, train_transforms)
        val_dataset_3d = MonaiAugmentedDataset(val_dataset_3d, noaugtransforms)
        test_dataset_3d = MonaiAugmentedDataset(test_dataset_3d, noaugtransforms)
        return (
            DataLoader(train_dataset_3d, batch_size=batch_size, shuffle=True, num_workers=num_workers),
            DataLoader(val_dataset_3d, batch_size=batch_size, shuffle=False, num_workers=num_workers),
            DataLoader(test_dataset_3d, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        )
    
    print(len(train_dataset_3d), len(val_dataset_3d), len(test_dataset_3d))

    # 4 - convert the 3d images to a 2d axial slice dataset
    datasets_2d = [MRISegDataset2DFrom3D(ds, transforms=None) for ds in [train_dataset_3d, val_dataset_3d, test_dataset_3d]]

    # 5 - remove a proportion of axial slices with no label
    # train_dataset, val_dataset, test_dataset = [FilteredEmptyElementsDataset(ds, seed=seed, transforms=None, empty_proportion_retained=empty_proportion_retained) for ds in datasets_2d]

    train_dataset = MonaiAugmentedDataset(datasets_2d[0], train_transforms)
    val_dataset = MonaiAugmentedDataset(datasets_2d[1], noaugtransforms)
    test_dataset = MonaiAugmentedDataset(datasets_2d[2], noaugtransforms)
    
    if merge_val_test.lower() == 'true':
            val_dataset = ConcatDataset([val_dataset, test_dataset])

    # 6 - create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    if merge_val_test.lower() != 'true':
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
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
