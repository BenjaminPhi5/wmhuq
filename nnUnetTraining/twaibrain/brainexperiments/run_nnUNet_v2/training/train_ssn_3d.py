print("loading imports...")
# base imports
import torch
import torch.nn as nn
import numpy as np
import os
from torchinfo import summary
from collections import defaultdict, namedtuple
import argparse

# model architecture
import json
from twaibrain.braintorch.models.nnUNet.nnUNetV2_model_loader import get_network_from_plans
from twaibrain.braintorch.models.ssn import SSN_Wrapped_Deep_Supervision, SSN_Wrapped_Deep_Supervision_LLO, Hierarchical_SSN_with_ConvRefine, Hierarchical_SSN_with_ConvSpatialAttention

# fitting code
from twaibrain.braintorch.fitting_and_inference.get_trainer import get_trainer
from twaibrain.braintorch.fitting_and_inference.get_scratch_dir import scratch_dir
from twaibrain.braintorch.fitting_and_inference.optimizer_constructor import OptimizerConfigurator
from twaibrain.braintorch.fitting_and_inference.lightning_fitter import StandardLitModelWrapper

# loss function
from twaibrain.braintorch.losses.ssn_losses_V2 import SSN_ComboLoss, SSNDeepSupervisionLoss, LLOMultiDeepSupervisionLoss
from twaibrain.braintorch.losses.xent import dice_xent_loss

# data
from twaibrain.brainexperiments.run_nnUNet_v2.old_dataloading.dataset_pipelines import load_data
from twaibrain.braintorch.data.legacy_dataset_types.dataset_wrappers import MonaiAugmentedDataset
from twaibrain.braintorch.augmentation.nnunet_augmentations import get_nnunet_transforms, get_val_transforms

print("imported")

VOXELS_TO_WMH_RATIO = 382
VOXELS_TO_WMH_RATIO_EXCLUDING_EMPTY_SLICES = 140

def construct_parser():
    parser = argparse.ArgumentParser(description = "train models")
    
    # folder arguments
    parser.add_argument('--ckpt_dir', default='s2208943/results/revamped_models/', type=str)
    
    # data generation arguments
    parser.add_argument('--dataset', default='ed', type=str)
    parser.add_argument('--seed', default=3407, type=int)
    parser.add_argument('--test_split', default=0.15, type=float)
    parser.add_argument('--val_split', default=0.15, type=float)
    parser.add_argument('--empty_slice_retention', default=0.1, type=float)
    parser.add_argument('--cv_split', default=0, type=int)
    parser.add_argument('--cv_test_fold_smooth', default=1, type=int)  
    parser.add_argument('--no_test_fold', default='false', type=str)
    
    # general arguments for the loss function
    parser.add_argument('--dice_factor', default=1, type=float) # 5
    parser.add_argument('--xent_factor', default=1, type=float) # 0.01
    parser.add_argument('--xent_weight', default="none", type=str)
    parser.add_argument('--dice_empty_slice_weight', default=0.5, type=float)
    parser.add_argument('--reduction', default='mean_sum', type=str)
    
    # ssn arguments
    parser.add_argument('--ssn_rank', default=25, type=int)
    parser.add_argument('--ssn_epsilon', default=1e-5, type=float)
    parser.add_argument('--ssn_mc_samples', default=10, type=int)
    parser.add_argument('--ssn_sample_dice_coeff', default=1, type=float)
    parser.add_argument('--ssn_pre_head_layers', default=32, type=int)
    parser.add_argument('--final_head_only', action='store_true')
    parser.add_argument('--spatial_attention_head', action='store_true')
    parser.add_argument('--upsampled_ssn', action='store_true')
    parser.add_argument('--mean_weight', type=float, default=1/11)
    parser.add_argument('--sample_weight', type=float, default=10/11)
    parser.add_argument('--dice_added_sample_weight', type=float, default=1)
    parser.add_argument('--best_sample', type=int, default=0, help="if 1, take loss from best sample only, otherwise take loss across all samples")
    
    # training paradigm arguments
    parser.add_argument('--max_epochs', default=1000, type=int)
    parser.add_argument('--early_stop_patience', default=100, type=int)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--batch_accumulation', default=1, type=int)
    parser.add_argument('--num_workers', default=6, type=int)
    parser.add_argument('--config_file', default=0, type=int)
    parser.add_argument('--nnUnet_augmentation', default=1, type=int, help="if 1 then use nnUnet augmentation, if 0 then use simple augmentation")
    
    return parser


def main(args):
    # load data
    print("loading data...")
    train_dl, val_dl, test_dl = load_data(
        dataset=args.dataset, 
        test_proportion=args.test_split, 
        validation_proportion=args.val_split,
        seed=args.seed,
        empty_proportion_retained=args.empty_slice_retention,
        batch_size=args.batch_size,
        dataloader3d_only=True,
        # dataset3d_only=True,
        # dataloader2d_only=True,
        cross_validate=True,
        cv_split=args.cv_split,
        cv_test_fold_smooth=args.cv_test_fold_smooth,
        merge_val_test=args.no_test_fold,
        num_workers=args.num_workers,
        nnUnet_augment=args.nnUnet_augmentation == 1,
    )

    # load model
    print("constructing model...")
    config_file_standard = "/home/s2208943/projects/twaibrain/twaibrain/braintorch/models/nnUNet/cvd_configs/old_nnUNetPlans_a40_standard_pl.json"
    config_file_M = "/home/s2208943/projects/twaibrain/twaibrain/braintorch/models/nnUNet/cvd_configs/nnUNetResEncUNetMPlans.json"
    
    model_config = config_file_standard if args.config_file == 0 else (config_file_M if args.config_file == 1 else None)
    if model_config is None:
        raise ValueError("config file must be 0 or 1")
    with open(model_config) as f:
        model_config = json.load(f)

    dims = "3d_fullres"
    config = model_config['configurations'][dims]['architecture']
    network_name = config['network_class_name']
    kw_requires_import = config['_kw_requires_import']
    
    model = get_network_from_plans(
        arch_class_name=network_name,
        arch_kwargs=config['arch_kwargs'],
        arch_kwargs_req_import=kw_requires_import,
        input_channels=3,
        output_channels=args.ssn_pre_head_layers,
        allow_init=True,
        deep_supervision=True,
    )
    
    ssn_config = {
        'intermediate_channels':args.ssn_pre_head_layers,
        'out_channels':2,
        'dims':3,
        'rank':args.ssn_rank,
        'diagonal':False,
    }

    if args.spatial_attention_head:
        ssn_model = Hierarchical_SSN_with_ConvSpatialAttention(model, ssn_config, refine_channels=10)
    elif args.upsampled_ssn:
        ssn_model = Hierarchical_SSN_with_ConvRefine(model, ssn_config, refine_channels=10)
    elif args.final_head_only:
        ssn_model =  SSN_Wrapped_Deep_Supervision_LLO(model, ssn_config)
    else:
        ssn_model = SSN_Wrapped_Deep_Supervision(model, 5, ssn_config)

    print("setting up optimization process")
    XENT_VOXEL_RESCALE = VOXELS_TO_WMH_RATIO - (1-args.empty_slice_retention) * (VOXELS_TO_WMH_RATIO - VOXELS_TO_WMH_RATIO_EXCLUDING_EMPTY_SLICES)

    XENT_WEIGHTING = XENT_VOXEL_RESCALE/2
    xent_reweighting = XENT_WEIGHTING

    combo_loss = SSN_ComboLoss(
        dice_weight=args.dice_factor, 
        xent_weight=args.xent_factor * xent_reweighting, 
        mean_weight=args.mean_weight, 
        sample_weight=args.sample_weight, 
        dice_added_sample_weight=args.dice_added_sample_weight,
        best_sample=args.best_sample==1, 
        mc_samples=args.ssn_mc_samples,
    )

    deterministic_combo_loss = dice_xent_loss(dice_weight=1, xent_weight=args.xent_factor *xent_reweighting)

    if args.spatial_attention_head or args.upsampled_ssn or args.final_head_only:
        loss = LLOMultiDeepSupervisionLoss(combo_loss, deterministic_combo_loss)
    else:
        loss = SSNDeepSupervisionLoss(combo_loss)
    
    torch.set_float32_matmul_precision('medium')
    optim_configurator = OptimizerConfigurator("SGD lr:0.01 momentum:0.99 nesterov:true", f"PolynomialLR total_iters:{args.max_epochs} power:0.9")
    litmodel = StandardLitModelWrapper(
        ssn_model,
        loss,
        optim_configurator,
        check_finiteness_of_data=False,
    )
    trainer = get_trainer(
        args.max_epochs,
        args.ckpt_dir,
        save_top_k=1,
        early_stop_patience=args.early_stop_patience,
        use_early_stopping=True,
        accumulate_grad_batches=args.batch_accumulation, 
        scheduled_accumuate_gradients=False,
        do_gradient_clip=True,
        gradient_clip_val=1
    )

    print("training model")
    trainer.fit(litmodel, train_dl, val_dl)

    print("DONE!")


if __name__ == '__main__':
    parser = construct_parser()
    args = parser.parse_args()
    main(args)
    
