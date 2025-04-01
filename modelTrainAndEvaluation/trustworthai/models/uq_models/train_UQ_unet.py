print("importing")

import torch
import numpy as np

# dataset
from twaidata.torchdatasets.in_ram_ds import MRISegmentation2DDataset, MRISegmentation3DDataset
from torch.utils.data import DataLoader, random_split, ConcatDataset

# model
from trustworthai.models.uq_models.drop_UNet import UNet

# augmentation and pretrain processing
from trustworthai.utils.augmentation.standard_transforms import RandomFlip, GaussianBlur, GaussianNoise, \
                                                            RandomResizeCrop, RandomAffine, \
                                                            NormalizeImg, PairedCompose, LabelSelect, \
                                                            PairedCentreCrop, CropZDim
# loss function
from trustworthai.utils.losses_and_metrics.tversky_loss import TverskyLoss
from trustworthai.utils.losses_and_metrics.misc_metrics import IOU
from trustworthai.utils.losses_and_metrics.dice import dice, DiceMetric
from trustworthai.utils.losses_and_metrics.dice_losses import DiceLoss, GeneralizedDiceLoss
from trustworthai.utils.losses_and_metrics.power_jaccard_loss import PowerJaccardLoss
from torch.nn import BCELoss, MSELoss, BCEWithLogitsLoss

# fitter
from trustworthai.utils.fitting_and_inference.fitters.basic_lightning_fitter import StandardLitModelWrapper
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
import pytorch_lightning as pl

# misc
import os
import argparse

def construct_parser():
    parser = argparse.ArgumentParser(description = "train standard UNet with various dropout and dropconnect layers - on cluster script")
    
    parser.add_argument('--dropout_type', default=None, type=str)
    parser.add_argument('--dropconnect_type', default=None, type=str)
    parser.add_argument('--dropout_p', default=None, type=float)
    parser.add_argument('--dropconnect_p', default=None, type=float)
    parser.add_argument('--norm_type', default='bn', type=str)
    parser.add_argument('--use_multidim_dropout', default=None, type=str)
    parser.add_argument('--use_multidim_dropconnect', default=None, type=str)
    parser.add_argument('--max_epochs', default=400, type=int)
    parser.add_argument('--gn_groups', default=4, type=int)
    parser.add_argument('--state_ckpt_and_exit', default=0, type=int)
    return parser

def extract_bool(b: str, argname):
    if b == None:
        return None
    b = b.lower()
    if b == "false":
        return False
    elif b == "true":
        return True
    else:
        raise ValueError(f"expect true or false, not {b} for arg {argname}")

def main(args):
    #======================================================================================================================#
    # CONSTS
    #======================================================================================================================#
    seed = 3407

    test_proportion = 0.1
    validation_proportion = 0.2

    checkpoint_dir = "/disk/scratch/s2208943/results/dropout_and_norm_initial_tests/"
    root_dir = "/disk/scratch/s2208943/ipdis/preprep/out_data/collated/"
    wmh_dir = root_dir + "WMH_challenge_dataset/"
    ed_dir = root_dir + "EdData/"

    # domains = [
    #             wmh_dir + d for d in ["Singapore", "Utrecht", "GE3T"]
    #           ]
    # domains = [
    #             wmh_dir + d for d in ["Singapore", "Utrecht", "GE3T"]
    #           ] + [
    #             ed_dir + d for d in ["domainA", "domainB", "domainC", "domainD"]
    #           ]
    domains = [
                ed_dir + d for d in ["domainA", "domainB", "domainC", "domainD"]
              ]


    is3D = False

    z_crop_size = 32 # how many slices to randomly use as input to the 3D model
    in_channels = 3 # brain mask, falir, t1
    out_channels = 1 # wmh label 
    label_id = 1 # which label we are interested in learning (label 1 is WMH in WMH challenge dataset)
    
    #======================================================================================================================#
    # EXTRACT ARGS
    #======================================================================================================================#
    # get out the bool arguments
    use_multidim_dropout = extract_bool(args.use_multidim_dropout, "use_multidim_dropout")
    use_multidim_dropconnect = extract_bool(args.use_multidim_dropconnect, "use_multidim_dropconnect")
    
    # calculate the checkpoint folder name
    dropout_str = f"dropout_{args.dropout_type}_{args.dropout_p}_" if args.dropout_type != None else ""
    
    a = use_multidim_dropout
    b = (args.dropout_type != None)
    
    if (use_multidim_dropout and (args.dropout_type != None)):
        dropout_str = f"{dropout_str}mdim_"
    
    dropconn_str = f"dropconn_{args.dropconnect_type}_{args.dropconnect_p}_" if args.dropconnect_type != None else ""
    if use_multidim_dropconnect and args.dropconnect_type != None:
        dropconn_str = f"{dropconn_str}mdim_"
    
    norm_str = args.norm_type
    epochs_str = f"epochs_{args.max_epochs}"
    
    checkpoint_dir = os.path.join(checkpoint_dir, f"UNet2D_{dropout_str}{dropconn_str}{norm_str}_{epochs_str}")
    print("checkpoint_dir: ", checkpoint_dir)
    
    if args.state_ckpt_and_exit == 1:
        return
    
    
    #======================================================================================================================#
    # SET SEED
    #======================================================================================================================#
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    #======================================================================================================================#
    # GET DATASET AND DATALOADER
    #======================================================================================================================#
    # augmentation definintion
    def get_transforms(is_3D):
        transforms = [
            LabelSelect(label_id=label_id),
            RandomFlip(p=0.5, orientation="horizontal"),
            # GaussianBlur(p=0.5, kernel_size=7, sigma=(.1, 1.5)),
            # GaussianNoise(p=0.2, mean=0, sigma=0.2),
            # RandomAffine(p=0.2, shear=(.1,3.)),
            # RandomAffine(p=0.2, degrees=5),
            # RandomResizeCrop(p=1., scale=(0.6, 1.), ratio=(3./4., 4./3.))
        ]
        if not is_3D:
            return PairedCompose(transforms)
        else:
            transforms.append(CropZDim(size=z_crop_size, minimum=0, maximum=-1))
            return PairedCompose(transforms)

    # function to do train validate test split
    def train_val_test_split(dataset, val_prop, test_prop, seed):
        # I think the sklearn version might be prefereable for determinism and things
        # but that involves fiddling with the dataset implementation I think....
        size = len(dataset)
        test_size = int(test_prop*size) 
        val_size = int(val_prop*size)
        train_size = size - val_size - test_size
        train, val, test = random_split(dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(seed))
        return train, val, test


    print("loading data")
    # load datasets
    # this step is quite slow, all the data is being loaded into memory
    if is3D:
        datasets_domains = [MRISegmentation3DDataset(root_dir, domain, transforms=get_transforms(is_3D=True)) for domain in domains]
    else:
        datasets_domains = [MRISegmentation2DDataset(root_dir, domain, transforms=get_transforms(is_3D=False)) for domain in domains]

    # split into train, val test datasets
    datasets = [train_val_test_split(dataset, validation_proportion, test_proportion, seed) for dataset in datasets_domains]

    # concat the train val test datsets
    train_dataset = ConcatDataset([ds[0] for ds in datasets])
    val_dataset = ConcatDataset([ds[1] for ds in datasets])
    test_dataset = ConcatDataset([ds[2] for ds in datasets])

    # define dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
    test_dataloader = DataLoader(test_dataset, batch_size = 16, shuffle=False, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)

    #======================================================================================================================#
    # SETUP MODEL
    #======================================================================================================================#
    print("loading model")
    if is3D:
        model = UNet3D(in_channels,
                     out_channels,
                     init_features=32,
                     kernel_size=3,
                     softmax=False,
                     dropout_type=None,
                     dropout_p=None,
                     gaussout_mean=None, 
                     dropconnect_type=None,
                     dropconnect_p=None,
                     gaussconnect_mean=None,
                     norm_type="bn", 
                     use_multidim_dropout = None,  
                     use_multidim_dropconnect = None, 
                     groups=None,
                     gn_groups=None, 
                    )
        optimizer_params={"lr":2e-3}
        lr_scheduler_params={"step_size":100, "gamma":0.5}
    else:
        model = UNet(in_channels,
                     out_channels,
                     kernel_size=3,
                     init_features=32,
                     softmax=False,
                     dropout_type=args.dropout_type,
                     dropout_p=args.dropout_p,
                     gaussout_mean=1, 
                     dropconnect_type=args.dropconnect_type,
                     dropconnect_p=args.dropconnect_p,
                     gaussconnect_mean=1,
                     norm_type=args.norm_type,
                     use_multidim_dropout=args.use_multidim_dropout,  
                     use_multidim_dropconnect=args.use_multidim_dropconnect, 
                     groups=None,
                     gn_groups=args.gn_groups, 
                    )
        optimizer_params={"lr":1e-3}
        lr_scheduler_params={"step_size":10, "gamma":0.1}


    loss = GeneralizedDiceLoss(normalization='sigmoid')
    model = StandardLitModelWrapper(model, loss, 
                                optimizer_params=optimizer_params,
                                lr_scheduler_params=lr_scheduler_params,
                                is_uq_model=True
                               )


    strategy = None
    # strategy = "deepspeed_stage_2"
    # strategy = "dp"
    #strategy = "deepspeed_stage_2_offload"

    accelerator="gpu"
    devices=1
    max_epochs=args.max_epochs
    precision = 16

    checkpoint_callback = ModelCheckpoint(checkpoint_dir, save_top_k=2, monitor="val_loss")
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=20, verbose="False", mode="min", check_finite=True)
    trainer = pl.Trainer(
        callbacks=[checkpoint_callback, early_stop_callback],
        accelerator=accelerator,
        devices=devices,
        max_epochs=max_epochs,
        strategy=strategy,
        precision=precision,
        default_root_dir=checkpoint_dir
    )

    #======================================================================================================================#
    # TRAIN
    #======================================================================================================================#
    print("training")
    trainer.fit(model, train_dataloader, val_dataloader)

    # checkpoints are saved automatically in the checkpoint folder you pick.
    
if __name__ == '__main__':
    parser = construct_parser()
    args = parser.parse_args()
    main(args)
