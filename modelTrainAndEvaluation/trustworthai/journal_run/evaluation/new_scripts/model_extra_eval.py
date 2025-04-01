"""
This in an extra set of analysis. We are doing voxelwise recall and f1, as well as lesion f1 and recall in the deep region and pv region separately
we are then computing a ROC cuve.

for the uncertainty analysis, we are going to do PVWMH and DWMH UEO with dice, recall and f1. 

so for this I need 

- [x] voxelwise recall func
- [x] voxelwise precision func
- [x] voxelwise tpr (recall)
- [x] voxelwise fpr
- [-] function to split into DWMH and PVWMH
- [-] function to do the lesion cluster analysis using the challenge code
- [x] modifications to the main func to run the new analysis
- [x] a new save file name for the evaluation, make sure it doesn't overwrite existing results!!!!
- [ ] relevant slurm scripts
- [x] UEO based on the recall and precision metrics
"""

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
    
    
    # check if folder exists
    model_result_folder = os.path.join(args.repo_dir, args.result_dir)
    if not args.overwrite:
        existing_files = os.listdir(model_result_folder)
        for f in existing_files:
            if args.model_name + "_extra_results_" in f:
                raise ValueError(f"ovewrite = false and model results exist! folder={model_result_folder}, model_name={args.model_name}")
    with open(os.path.join(model_result_folder, f"{args.model_name}_extra_results_init.txt"), "w") as f:
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

    chal_results = {}

    precisions = []
    recalls = []
    dices = []
    avds = []
    fprs = []
    for m, y in zip(means, ys3d_test):
        m = m.cuda()
        if "evid" not in args.uncertainty_type:
            m = m.softmax(dim=1)
        m = m.argmax(dim=1)
        print(m.shape)
        precisions.append(fast_precision(m, y.cuda()))
        recalls.append(fast_recall(m, y.cuda()))
        dices.append(fast_dice(m, y.cuda()))
        avds.append(fast_avd(m, y.cuda()))
        fprs.append(fast_fpr(m, y.cuda()))
    precisions = torch.Tensor(precisions)
    recalls = torch.Tensor(recalls)
    dices = torch.Tensor(dices)
    avds = torch.Tensor(avds)
    fprs = torch.Tensor(fprs)
    chal_results['voxelwise_precision'] = precisions
    chal_results['voxelwise_recall'] = recalls
    chal_results['V2dice'] = dices
    chal_results['V2avd'] = avds
    chal_results['voxelwise_fpr'] = fprs

    # run the evaluation on the samples
    print("GETTING EXTRA PER SAMPLE RESULTS")
    if samples[0] is not None:
        samples = [reorder_samples(s) for s in samples]
        sample_top_recall, sample_recall = per_sample_metric(samples, ys3d_test, f=fast_dice, do_argmax=True, do_softmax=False, minimise=False)
        sample_best_precision, sample_precision = per_sample_metric(samples, ys3d_test, f=fast_avd, do_argmax=True, do_softmax=False, minimise=True)
        
        # best dice, avd, rmse
        chal_results['best_recall'] = sample_top_recall
        chal_results['best_precision'] = sample_best_precision
            
        # ged score
        geds = iou_GED(means, ys3d_test, samples)
        chal_results['V2GED^2'] = geds
        
    # get the uncertainty maps
    print("GENREATING UNCERTAINTY MAPS")
    uncertainty_thresholds = torch.arange(0, 0.7, 0.01)
    ent_maps = get_uncertainty_maps(means, samples, misc, args)
    
    # equivalent versions for the precision and Recall
    print("UEO_recall and UEO_precision")
    ueos_recall, ueos_precision = UEO_per_threshold_recall_precision(uncertainty_thresholds, ys3d_test, ent_maps, means, 0.7)
    for i, tau in enumerate(uncertainty_thresholds):
        chal_results[f'UEOrecall_{tau:.2f}'] = ueos_recall[i]
        chal_results[f'UEOprecision_{tau:.2f}'] = ueos_precision[i]
        
    # get the information for the ROC curve
    print("ROC curve data")
    threshold_recalls = defaultdict(lambda : [])
    threshold_precision = defaultdict(lambda : [])
    for m, y in zip(means, ys3d_test):
        m = m.cuda()
        if "evid" not in args.uncertainty_type:
            m = m.softmax(dim=1)
        y = y.cuda()
        for i, tau in enumerate(np.arange(0, 1, 0.01)):
            mt = m[:,1] >= tau
            threshold_recalls[f'voxelwise_trecall_{tau:.2f}'].append(fast_recall(mt, y))
            threshold_precision[f'voxelwise_tprecision_{tau:.2f}'].append(fast_precision(mt, y))
    for i, tau in enumerate(np.arange(0, 1, 0.01)):
        chal_results[f'voxelwise_trecall_{tau:.2f}'] = torch.tensor(threshold_recalls[f'voxelwise_trecall_{tau:.2f}'])
        chal_results[f'voxelwise_tprecision_{tau:.2f}'] = torch.tensor(threshold_precision[f'voxelwise_tprecision_{tau:.2f}'])
        
    # save the results
    for key in chal_results.keys():
        print(key, len(chal_results[key]), type(chal_results[key]))
        
    print("SAVING RESULTS")
    write_per_model_channel_stats(preds=None, ys3d_test=None, args=args, chal_results=chal_results, fname='overall_extra_stats.csv', individual_stats_name='individual_extra_stats.csv')
    
    print("DONE")

if __name__ == '__main__':
    parser = construct_parser()
    args = parser.parse_args()
    main(args)
