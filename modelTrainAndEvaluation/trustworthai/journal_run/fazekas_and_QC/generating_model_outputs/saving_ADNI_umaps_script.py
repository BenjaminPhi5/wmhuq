print("strawberry")

# loss function and metrics
from trustworthai.utils.losses_and_metrics.dice_loss import DiceLossWithWeightedEmptySlices
from trustworthai.utils.losses_and_metrics.dice_loss_metric import DiceLossMetric, SsnDiceMeanMetricWrapper

# predefined training dataset
from trustworthai.utils.data_preprep.dataset_pipelines import load_data
from torch.utils.data import ConcatDataset

# fitter
from trustworthai.utils.fitting_and_inference.fitters.basic_lightning_fitter import StandardLitModelWrapper
from trustworthai.utils.fitting_and_inference.fitters.p_unet_fitter import PUNetLitModelWrapper
from trustworthai.utils.fitting_and_inference.get_trainer import get_trainer

# model
from trustworthai.journal_run.model_load.load_ssn import load_ssn
from trustworthai.journal_run.model_load.load_punet import load_p_unet
from trustworthai.journal_run.model_load.load_deterministic import load_deterministic
from trustworthai.journal_run.model_load.load_evidential import load_evidential
from trustworthai.models.stochastic_wrappers.ssn.LowRankMVCustom import LowRankMultivariateNormalCustom
from trustworthai.models.stochastic_wrappers.ssn.ReshapedDistribution import ReshapedDistribution

# optimizer and lr scheduler
import torch


import numpy as np
from tqdm import tqdm
import scipy.stats
from trustworthai.utils.plotting.saving_plots import save, imsave
from trustworthai.utils.print_and_write_func import print_and_write

# misc
import argparse
import os
import shutil
import shlex
from collections import defaultdict
from tqdm import tqdm
import sys
from natsort import natsorted

import pandas as pd
from trustworthai.analysis.connected_components.connected_comps_2d import conn_comp_2d_analysis
from trustworthai.analysis.evaluation_metrics.challenge_metrics import getAVD, getDSC, getHausdorff, getLesionDetection, do_challenge_metrics
from sklearn import metrics
import math

import torch
import matplotlib.pyplot as plt
from trustworthai.utils.plotting.saving_plots import save
from trustworthai.utils.print_and_write_func import print_and_write
from trustworthai.analysis.calibration.helper_funcs import *
from tqdm import tqdm
from trustworthai.utils.logits_to_preds import normalize_samples

# data
from trustworthai.utils.uncertainty_maps.entropy_map import entropy_map_from_samples
# from twaidata.torchdatasets.MRI_3D_nolabels_inram_ds import MRI_3D_nolabels_inram_ds # old code
from twaidata.torchdatasets_v2.mri_dataset_inram import MRISegmentation3DDataset

# evaluation code
from trustworthai.journal_run.evaluation.new_scripts.model_predictions import *
from trustworthai.journal_run.new_MIA_fazekas_and_QC.generating_model_outputs.utils import *

print("banana")

MODEL_LOADERS = {
    "deterministic":load_deterministic,
    "mc_drop":load_deterministic,
    "evidential":load_evidential,
    "ssn":load_ssn,
    "punet":load_p_unet,
}

MODEL_OUTPUT_GENERATORS = {
    "deterministic":deterministic_mean,
    "mc_drop":mc_drop_mean_and_samples,
    "evidential":evid_mean,
    "ssn":ssn_mean_and_samples,
    "punet":punet_mean_and_samples,
    "ind":ssn_mean_and_samples,
    "ens":ensemble_mean_and_samples,
    "ssn_ens":ssn_ensemble_mean_and_samples,
}

VOXELS_TO_WMH_RATIO = 382
VOXELS_TO_WMH_RATIO_EXCLUDING_EMPTY_SLICES = 140

def construct_parser():
    parser = argparse.ArgumentParser(description = "train models")
    
    # folder arguments
    parser.add_argument('--ckpt_dir', default='s2208943/results/revamped_models/', type=str)
    parser.add_argument('--model_name', default=None, type=str)
    parser.add_argument('--repo_dir', default=None, type=str)
    parser.add_argument('--result_dir', default=None, type=str)
    parser.add_argument('--eval_split', default='val', type=str)
    
    # data generation arguments
    parser.add_argument('--dataset', default='ed', type=str)
    parser.add_argument('--seed', default=3407, type=int)
    parser.add_argument('--test_split', default=0.15, type=float)
    parser.add_argument('--val_split', default=0.15, type=float)
    parser.add_argument('--empty_slice_retention', default=0.1, type=float)
    
    # select the model type to evaluate
    parser.add_argument('--model_type', default="deterministic", type=str)
    parser.add_argument('--uncertainty_type', default="deterministic", type=str)
    parser.add_argument('--eval_sample_num', default=10, type=int)
    
    # general arguments for the loss function
    parser.add_argument('--loss_name', default='dice+xent', type=str)
    parser.add_argument('--dice_factor', default=1, type=float) # 5
    parser.add_argument('--xent_factor', default=1, type=float) # 0.01
    parser.add_argument('--xent_reweighting', default=None, type=float)
    parser.add_argument('--xent_weight', default="none", type=str)
    parser.add_argument('--dice_empty_slice_weight', default=0.5, type=float)
    parser.add_argument('--tversky_beta', default=0.7, type=float)
    parser.add_argument('--reduction', default='mean_sum', type=str)
    
    # evidential arguments
    parser.add_argument('--kl_factor', default=0.1, type=float)
    parser.add_argument('--kl_anneal_count', default=452*4, type=int)
    parser.add_argument('--use_mle', default=0, type=int)
    parser.add_argument('--analytic_kl', default=0, type=int)
    
    # p-unet arguments
    parser.add_argument('--kl_beta', default=10.0, type=float)
    parser.add_argument('--use_prior_for_dice', default="false", type=str)
    parser.add_argument('--punet_sample_dice_coeff', default=0.05, type=float)
    parser.add_argument('--latent_dim', default=12, type=int)
    
    # ssn arguments
    parser.add_argument('--ssn_rank', default=15, type=int)
    parser.add_argument('--ssn_epsilon', default=1e-5, type=float)
    parser.add_argument('--ssn_mc_samples', default=10, type=int)
    parser.add_argument('--ssn_sample_dice_coeff', default=0.05, type=float)
    parser.add_argument('--ssn_pre_head_layers', default=16, type=int)
    
    # training paradigm arguments
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--dropout_p', default=0.0, type=float)
    parser.add_argument('--encoder_dropout1', default=0, type=int)
    parser.add_argument('--encoder_dropout2', default=0, type=int)
    parser.add_argument('--decoder_dropout1', default=0, type=int)
    parser.add_argument('--decoder_dropout2', default=0, type=int)
    parser.add_argument('--max_epochs', default=100, type=int)
    parser.add_argument('--early_stop_patience', default=15, type=int)
    parser.add_argument('--scheduler_type', default='step', type=str)
    parser.add_argument('--scheduler_step_size', default=1000, type=int)
    parser.add_argument('--scheduler_gamma', default=0.5, type=float)
    parser.add_argument('--scheduler_power', default=0.9, type=float)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--cross_validate', default="true", type=str)
    parser.add_argument('--cv_split', default=0, type=int)
    parser.add_argument('--cv_test_fold_smooth', default=1, type=int)
    parser.add_argument('--weight_decay', default=0.0001, type=float)
    parser.add_argument('--overwrite', default="false", type=str)
    parser.add_argument('--no_test_fold', default='false', type=str)
    
    return parser

def main(args):
    # sanitise arguments
    args.overwrite = True if args.overwrite.lower() == "true" else False
    args.cross_validate = True if args.cross_validate.lower() == "true" else False
    args.use_prior_for_dice = True if args.use_prior_for_dice.lower() == "true" else False
    print(f"CHECKPOINT DIR: {args.ckpt_dir}")
    #print(args)
    
    
    # check if folder exists
    model_result_folder = os.path.join(args.result_dir, args.uncertainty_type)
    if args.overwrite:
        if os.path.exists(model_result_folder):
            raise ValueError(f"ovewrite = false and model results exist! folder={model_result_folder}")
    
    # setup xent reweighting factor
    XENT_VOXEL_RESCALE = VOXELS_TO_WMH_RATIO - (1-args.empty_slice_retention) * (VOXELS_TO_WMH_RATIO - VOXELS_TO_WMH_RATIO_EXCLUDING_EMPTY_SLICES)

    XENT_WEIGHTING = XENT_VOXEL_RESCALE/2
    args.xent_reweighting = XENT_WEIGHTING
    
    # load the adni data
    # ADNI_ds = MRI_3D_nolabels_inram_ds("/home/s2208943/ipdis/data/preprocessed_data/ADNI_300") # old code from before I redid the data code
    ADNI_ds = MRISegmentation3DDataset("/home/s2208943/preprocessed_data/ADNI300/collated", no_labels=True, xy_only=False)
    
    # load the model
    print("LOADING INITIAL MODEL")
    model_dir = os.path.join(args.ckpt_dir, args.model_name)  
    print("model dir: ", model_dir)
    model_raw, loss, val_loss = MODEL_LOADERS[args.model_type](args)
    model = load_best_checkpoint(model_raw, loss, model_dir, punet=args.model_type == "punet")
        
    start = 0
    end = 10
    while end < 300:
        ds = LimitedSizeDataset(ADNI_ds, start=start, end=end)
        IDs = ADNI_ds[start:end][2]
        # IDs = ds[:][2]

        start = end
        end += 10

        # get predictions
        means, samples, misc = get_means_and_samples(model_raw, ds, num_samples=10, model_func=MODEL_OUTPUT_GENERATORS[args.uncertainty_type], args=args)

        # get uncertainty maps
        ent_maps = get_uncertainty_maps(means, samples, misc, args)

        # write results
        write_results_to_disk(args, IDs, ent_maps, means, output_dir=args.result_dir)

if __name__ == '__main__':
    parser = construct_parser()
    args = parser.parse_args()
    main(args)
