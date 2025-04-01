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
from trustworthai.utils.data_preprep.dataset_pipelines import load_data, ClinScoreDataRetriever
from trustworthai.utils.uncertainty_maps.entropy_map import entropy_map_from_samples


# evaluation code
from trustworthai.journal_run.evaluation.new_scripts.eval_helper_functions import *
from trustworthai.journal_run.evaluation.new_scripts.model_predictions import *
from trustworthai.analysis.connected_components.connected_comps_2d import *
from trustworthai.journal_run.interrater_experiments.interrater_eval_helper_funcs import *

from torch.utils.data import ConcatDataset
from twaidata.torchdatasets_v2.individual_dataset_wrappers import *

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

uncertainty_thresholds = torch.arange(0, 0.7, 0.01)


def construct_parser():
    parser = argparse.ArgumentParser(description = "train models")
    
    # folder arguments
    parser.add_argument('--ckpt_dir', default='s2208943/results/revamped_models/', type=str)
    parser.add_argument('--model_name', default=None, type=str)
    parser.add_argument('--repo_dir', default=None, type=str)
    parser.add_argument('--result_dir', default=None, type=str)
    parser.add_argument('--eval_split', default='val', type=str)
    
    # data generation arguments
    parser.add_argument('--dataset', default='MSS3', type=str)
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
    
    if args.dataset == "chal" and args.eval_split == "all":
        args.cv_split = 0 #  we are evaluating on the whole dataset anyway and the dataset doesn't divide into more than 5 folds with the parameters used on the ed dataset.
    #print(args)
    
    uncertainty_thresholds = torch.arange(0, 0.7, 0.01)
    
    # check if folder exists
    # model_result_folder = os.path.join(args.repo_dir, args.result_dir)
    # if not args.overwrite:
    #     existing_files = os.listdir(model_result_folder)
    #     for f in existing_files:
    #         if args.model_name + "_" in f:
    #             raise ValueError(f"ovewrite = false and model results exist! folder={model_result_folder}, model_name={args.model_name}")
    # with open(os.path.join(model_result_folder, f"{args.model_name}_init.txt"), "w") as f:
    #                       f.write("generating results\n")
        
    # setup xent reweighting factor
    XENT_VOXEL_RESCALE = VOXELS_TO_WMH_RATIO - (1-args.empty_slice_retention) * (VOXELS_TO_WMH_RATIO - VOXELS_TO_WMH_RATIO_EXCLUDING_EMPTY_SLICES)

    XENT_WEIGHTING = XENT_VOXEL_RESCALE/2
    args.xent_reweighting = XENT_WEIGHTING
    
    # load the model
    print("LOADING INITIAL MODEL")
    model_dir = os.path.join(args.ckpt_dir, args.model_name)  
    print("model dir: ", model_dir)
    model_raw, loss, val_loss = MODEL_LOADERS[args.model_type](args)
    model = load_best_checkpoint(model_raw, loss, model_dir, punet=args.model_type == "punet")
    
    # LOAD THE XS, YS AND PV REGION MASKS
    xs3d_test = []
    ys3d_test = []
    pv_region_masks = []

    ds_name = args.dataset.lower()
    if ds_name == "mss3":
        r0_id = "wmhes"
        r1_id = "wmhmvh"
        ds_ir_folder = "MSS3_InterRaterData"
        ds = MSS3InterRaterDataset()
    elif ds_name == "lbc":
        r0_id = "wmh"
        r1_id = "wmh_flthresh"
        ds_ir_folder = "LBC_InterRaterData"
        ds = LBCInterRaterDataset()
    elif ds_name == "challenge":
        r0_id = "wmho3"
        r1_id = "wmho4"
        ds_ir_folder = "WMHChallenge_InterRaterData"
        ds = WMHChallengeInterRaterDataset()
    else:
        raise ValueError(f"dataset name {ds_name} unknown")


    for (xs, ys, ind, _) in tqdm(ds):
        if ind in ['MSS3_ED_073_V1', 'MSS3_ED_075_V1', 'MSS3_ED_078_V1', 'MSS3_ED_079_V1']: # get rid of mss3 files with flipped images
            print("found")
            continue
        if r0_id in ys.keys() and r1_id in ys.keys():
            try:
                x = torch.stack([xs['FLAIR'], xs['T1'], xs['mask']], dim=0)
                y = torch.stack([ys[r0_id], ys[r1_id]], dim=0)

                vent_distance_map = xs['vent_distance']
            except:
                print(f"failed for {ind}")
                continue
                
            if x.shape[1] != vent_distance_map.shape[0] or x.shape[2] != vent_distance_map.shape[1] or x.shape[3] != vent_distance_map.shape[2]:
                continue

            xs3d_test.append(x)
            ys3d_test.append(y)
            pv_region_masks.append((vent_distance_map < 10).type(torch.float32))
    
    # configuring raters
    rater0 = [y[0] for y in ys3d_test]
    rater1 = [y[1] for y in ys3d_test]
    gt_vols = [(torch.sum(y[0]).item(), torch.sum(y[1]).item()) for y in ys3d_test]
    mean_gt_vols = [(torch.sum(y[0]).item() + torch.sum(y[1]).item())/2 for y in ys3d_test]
    
    rater0_ds = list(zip(xs3d_test, rater0))

    ### results collection loop below
    raters = [rater0, rater1]
    rater_results = [defaultdict(lambda : {}) for _ in range(len(raters))]
    overall_results = defaultdict(lambda: {})
    pixelwise_and_cc_results = defaultdict(lambda: {})

    print("loading model predictions")
    ns_init = 10 #if args.uncertainty_type == "ens" else 30
    means, samples_all, misc = get_means_and_samples(model_raw, rater0_ds, num_samples=ns_init, model_func=MODEL_OUTPUT_GENERATORS[args.uncertainty_type], args=args)

    rmses, IR_rmses = get_rmse_stats(means, rater0, rater1)

    all_result_ns = [2, 10, 30]
    
    for num_samples in [10]:
        overall_results[num_samples][f'rmses'] = rmses
        overall_results[num_samples][f'IR_rmses'] = IR_rmses
        print("NUM SAMPLES: ", num_samples)
        args.eval_sample_num = num_samples

        # load the predictions
        print("extracting sample subset")
        print(args.uncertainty_type)
        samples = [s[:num_samples] for s in tqdm(samples_all)]
        print("computing uncertainty maps")
        ent_maps = get_uncertainty_maps(means, samples, misc, args)

        print("collecting the metrics")
        all_dJUEOs = []
        all_dUIROs = []
        all_sdJUEOs = []
        all_sdUIROs = []
        for i in tqdm(range(len(means))):
            pred = means[i].cuda().argmax(dim=1)
            umap = ent_maps[i].cuda()
            y0 = rater0[i].cuda()
            y1 = rater1[i].cuda()
            sdJUEOs, sdUIROs, dJUEOs, dUIROs, distance_thresholds = perform_d_UEO_analysis(pred, umap, y0, y1, uncertainty_thresholds)
            all_dJUEOs.append(dJUEOs)
            all_dUIROs.append(dUIROs)
            all_sdJUEOs.append(sdJUEOs)
            all_sdUIROs.append(sdUIROs)
            
        all_dJUEOs = torch.Tensor(all_dJUEOs)
        all_dUIROs = torch.Tensor(all_dUIROs)
        all_sdJUEOs = torch.Tensor(all_sdJUEOs)
        all_sdUIROs = torch.Tensor(all_sdUIROs)
        
        for di, d in enumerate(distance_thresholds):
            overall_results[num_samples][f'soft_dUIRO_d{d:.2f}'] = all_sdUIROs[:, di]
            overall_results[num_samples][f'soft_dJUEO_d{d:.2f}'] = all_sdJUEOs[:, di]
            for ti, t in enumerate(uncertainty_thresholds):
                overall_results[num_samples][f'dUIRO_curves_d{d:.2f}_t{t:.2f}'] = all_dUIROs[:,di,ti]
                overall_results[num_samples][f'dJUEO_curves_d{d:.2f}_t{t:.2f}'] = all_dJUEOs[:,di,ti]
                
        overall_results[num_samples]['mean_gt_vols'] = mean_gt_vols
                
        for key, arr in overall_results[10].items():
            print(key, " : ", len(arr))
            
        path = "/home/s2208943/ipdis/WMH_UQ_assessment/trustworthai/journal_run/interrater_experiments/distance_based_overlap_results/"
        # for key in overall_results[num_samples].keys():
            # print(key, len(overall_results[num_samples][key]))
        pd.DataFrame(overall_results[num_samples]).to_csv(path + f"dsb_overlap_results_{args.dataset}_{args.uncertainty_type}_cv{args.cv_split}_ns{num_samples}.csv")


if __name__ == '__main__':
    parser = construct_parser()
    args = parser.parse_args()
    main(args)
