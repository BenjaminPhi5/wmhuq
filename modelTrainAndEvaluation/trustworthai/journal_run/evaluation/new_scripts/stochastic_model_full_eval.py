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


from trustworthai.utils.data_preprep.dataset_pipelines import load_clinscores_data, load_data, ClinScoreDataRetriever

from torch.utils.data import ConcatDataset

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
    
    # if args.dataset == "chal" and args.eval_split == "all":
    #     args.cv_split = 0 #  we are evaluating on the whole dataset anyway and the dataset doesn't divide into more than 5 folds with the parameters used on the ed dataset.
    #print(args)
    
    
    # check if folder exists
    model_result_folder = os.path.join(args.repo_dir, args.result_dir)
    if not args.overwrite:
        existing_files = os.listdir(model_result_folder)
        for f in existing_files:
            if args.model_name + "_" in f:
                raise ValueError(f"ovewrite = false and model results exist! folder={model_result_folder}, model_name={args.model_name}")
    with open(os.path.join(model_result_folder, f"{args.model_name}_init.txt"), "w") as f:
                          f.write("generating results\n")
        
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
        
    # get the 3d axial slice dataloaders
    data_dict = load_data(
        dataset=args.dataset, 
        test_proportion=args.test_split, 
        validation_proportion=args.val_split,
        seed=args.seed,
        empty_proportion_retained=args.empty_slice_retention,
        batch_size=args.batch_size,
        dataloader2d_only=False,
        cross_validate=args.cross_validate,
        cv_split=args.cv_split,
        cv_test_fold_smooth=args.cv_test_fold_smooth if args.dataset == "ed" else 0,
        merge_val_test=args.no_test_fold
    )
    
    if args.eval_split == "all":
        eval_ds = ConcatDataset([data_dict['train_dataset3d'], data_dict['val_dataset3d'], data_dict['test_dataset3d']])
    else:
        eval_ds = data_dict[f'{args.eval_split}_dataset3d']
    
    # get data as lists
    xs3d_test, ys3d_test = get_xs_and_ys(eval_ds)
    ys3d_test = [y * (y==1).type(y.dtype) for y in ys3d_test] # fix bug with challenge data having 3 classes on cluster only?
    gt_vols = GT_volumes(ys3d_test)
    
    # load the predictions
    print("GENERATING PREDICTIONS")
    print(args.uncertainty_type) 
    means, samples, misc = get_means_and_samples(model_raw, eval_ds, num_samples=10, model_func=MODEL_OUTPUT_GENERATORS[args.uncertainty_type], args=args)

    # run the evaluation on the means
    print("GETTING CHALLENGE RESULTS")
    chal_results = per_model_chal_stats(means, ys3d_test)

    rmses = []
    for m, y in zip(means, ys3d_test):
        m = m.cuda()
        if "evid" not in args.uncertainty_type:
            m = m.softmax(dim=1)
        rmses.append(fast_rmse(m, y.cuda()).cpu())
    rmses = torch.Tensor(rmses)
    chal_results['rmse'] = rmses

    chal_results['gt_vols'] = gt_vols

    # run the evaluation on the samples
    print("GETTING PER SAMPLE RESULTS")
    if samples[0] is not None:
        samples = [reorder_samples(s) for s in samples]
        sample_top_dices, sample_dices = per_sample_metric(samples, ys3d_test, f=fast_dice, do_argmax=True, do_softmax=False, minimise=False)
        sample_best_avds, sample_avds = per_sample_metric(samples, ys3d_test, f=fast_avd, do_argmax=True, do_softmax=False, minimise=True)
        sample_best_rmses, sample_rmses = per_sample_metric(samples, ys3d_test, f=fast_rmse, do_argmax=False, do_softmax=True, minimise=True)

        # best dice, avd, rmse
        chal_results['best_dice'] = sample_top_dices
        chal_results['best_avd'] = sample_best_avds
        chal_results['best_rmse'] = sample_best_rmses
        
        _, sample_vds = per_sample_metric(samples, ys3d_test, f=fast_vd, do_argmax=True, do_softmax=False, minimise=True, take_abs=False)
        sample_vd_skew = torch.from_numpy(scipy.stats.skew(sample_vds, axis=1, bias=True))

        # vd of the sample distribution
        chal_results['sample_vd_skew'] = sample_vd_skew
        for s in range(sample_vds.shape[1]):
            chal_results[f'sample_{s}_vd'] = sample_vds[:,s]
            
        # ged score
        geds = iou_GED(means, ys3d_test, samples)
        chal_results['GED^2'] = geds
        
    # get the uncertainty maps
    print("GENREATING UNCERTAINTY MAPS")
    uncertainty_thresholds = torch.arange(0, 0.7, 0.01)
    ent_maps = get_uncertainty_maps(means, samples, misc, args)
    
    # pavpu
    print("PAVPU")
    all_acc_cert, all_uncert_inacc,all_pavpu = all_individuals_pavpu(means, ent_maps, ys3d_test, 4, 0.8, uncertainty_thresholds)
    
    for i, tau in enumerate(uncertainty_thresholds):
        chal_results[f'p_acc_cert_{tau:.2f}'] = all_acc_cert[:,i]
        chal_results[f'p_uncert_inacc_{tau:.2f}'] = all_uncert_inacc[:,i]
        chal_results[f'pavpu_{tau:.2f}'] = all_pavpu[:,i]
    
    # sUEO score and UEO per threshold
    print("UEO")
    sUEOs = get_sUEOs(means, ys3d_test, ent_maps)
    chal_results['sUEO'] = sUEOs
    ueos = UEO_per_threshold_analysis(uncertainty_thresholds, ys3d_test, ent_maps, means, 0.7)
    for i, tau in enumerate(uncertainty_thresholds):
        chal_results[f'UEO_{tau:.2f}'] = ueos[i]
    
    # 3D connected component analysis
    print("3D CC ANALYSIS")
    num_lesions_all, sizes_all, mean_missed_area3d_all, mean_size_missed_lesions3d_all, mean_cov_mean_missed_lesions3d_all, prop_lesions_missed3d_all = do_3d_cc_analysis_per_individual(means, ys3d_test, ent_maps, uncertainty_thresholds)
    for i, tau in enumerate(uncertainty_thresholds):
        chal_results[f'mean_missed_area3d_all_{tau:.2f}'] = torch.stack(mean_missed_area3d_all)[:,i]
        chal_results[f'mean_cov_mean_missed_lesions3d_all_{tau:.2f}'] = torch.stack(mean_cov_mean_missed_lesions3d_all)[:,i]
        chal_results[f'mean_size_missed_lesions3d_all_{tau:.2f}'] = torch.stack(mean_size_missed_lesions3d_all)[:,i]
        chal_results[f'prop_lesions_missed3d_all_{tau:.2f}'] = torch.stack(prop_lesions_missed3d_all)[:,i]
        
    # save the results
    print("SAVING RESULTS")
    write_per_model_channel_stats(preds=None, ys3d_test=None, args=args, chal_results=chal_results)
    
    print("DONE")

if __name__ == '__main__':
    parser = construct_parser()
    args = parser.parse_args()
    main(args)
