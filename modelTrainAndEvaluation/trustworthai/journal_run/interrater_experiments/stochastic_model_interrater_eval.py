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


from trustworthai.utils.data_preprep.dataset_pipelines import load_clinscores_data, load_data, ClinScoreDataRetriever

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
    ns_init = 10 if args.uncertainty_type == "ens" else 30
    means, samples_all, misc = get_means_and_samples(model_raw, rater0_ds, num_samples=ns_init, model_func=MODEL_OUTPUT_GENERATORS[args.uncertainty_type], args=args)

    rmses, IR_rmses = get_rmse_stats(means, rater0, rater1)

    all_result_ns = [10]
    
    for num_samples in [2, 3, 5, 7, 10, 15, 20, 25, 30]:
        overall_results[num_samples][f'rmses'] = rmses
        overall_results[num_samples][f'IR_rmses'] = IR_rmses
        print("NUM SAMPLES: ", num_samples)
        args.eval_sample_num = num_samples
        try:
            # load the predictions
            print("extracting sample subset")
            print(args.uncertainty_type)
            samples = [s[:num_samples] for s in tqdm(samples_all)]
            print("computing uncertainty maps")
            ent_maps = get_uncertainty_maps(means, samples, misc, args)

            # run the evaluation on the samplesUIRO_curves
            print("GETTING PER SAMPLE RESULTS")
            if samples[0] is not None:
                samples = [reorder_samples(s) for s in samples]
                for r, rater in enumerate(raters):
                    sample_top_dices, sample_dices = per_sample_metric(samples, rater, f=fast_dice, do_argmax=True, do_softmax=False, minimise=False)
                    sample_best_avds, sample_avds = per_sample_metric(samples, rater, f=fast_avd, do_argmax=True, do_softmax=False, minimise=True)
                    # sample_best_rmses, sample_rmses = per_sample_metric(samples, ys3d_test, f=fast_rmse, do_argmax=False, do_softmax=True, minimise=True)

                    # rater_results[r][num_samples]['sample_top_dice'] = sample_top_dices
                    # rater_results[r][num_samples]['sample_best_avd'] = sample_best_avds
                    overall_results[num_samples][f'rater{r}_sample_top_dice'] = sample_top_dices
                    overall_results[num_samples][f'rater{r}_sample_best_avd'] = sample_best_avds

                # ged by volume
                print("COMPUTING GED BY VOLUME")
                overall_results[num_samples]['GED_vol_sorted'] = multirater_iou_GED(means, raters, samples)

                # UEO metrics
                if num_samples in all_result_ns:
                    print("UEO metrics curves")
                    uiro_curves, jueo_curves = per_threshold_ueos(means, ent_maps, rater0, rater1, xs3d_test)
                    uiro_curves = torch.Tensor(uiro_curves)
                    jueo_curves = torch.Tensor(jueo_curves)
                    for ti, t in enumerate(uncertainty_thresholds):
                        overall_results[num_samples][f'UIRO_curves_t{t:.2f}'] = uiro_curves[:,ti]
                        overall_results[num_samples][f'JUEO_curves_t{t:.2f}'] = jueo_curves[:,ti]

                    no_edge_uiro_curves, no_edge_jueo_curves = per_threshold_edge_deducted_ueos(means, ent_maps, rater0, rater1, xs3d_test)
                    no_edge_uiro_curves = torch.Tensor(no_edge_uiro_curves)
                    no_edge_jueo_curves = torch.Tensor(no_edge_jueo_curves)
                    for ti, t in enumerate(uncertainty_thresholds):
                        overall_results[num_samples][f'no_edge_uiro_curves_t{t:.2f}'] = no_edge_uiro_curves[:,ti]
                        overall_results[num_samples][f'no_edge_jueo_curves_t{t:.2f}'] = no_edge_jueo_curves[:,ti]

                    print("soft UEO metrics values")
                    sUIRO, sJUEO, sUEO_r1, sUEO_r2, s_ed_UIRO, s_ed_JUEO, deep_sUIRO, deep_sJUEO, pv_sUIRO, pv_sJUEO = soft_ueo_metrics(means, ent_maps, rater0, rater1, xs3d_test, pv_region_masks)
                    overall_results[num_samples]['sUIRO'] = sUIRO
                    overall_results[num_samples]['sJUEO'] = sJUEO
                    overall_results[num_samples]['sUEO_r1'] = sUEO_r1
                    overall_results[num_samples]['sUEO_r2'] = sUEO_r2
                    overall_results[num_samples]['s_ed_UIRO'] = s_ed_UIRO
                    overall_results[num_samples]['s_ed_JUEO'] = s_ed_JUEO
                    overall_results[num_samples]['deep_sUIRO'] = deep_sUIRO
                    overall_results[num_samples]['deep_sJUEO'] = deep_sJUEO
                    overall_results[num_samples]['pv_sUIRO'] = pv_sUIRO
                    overall_results[num_samples]['pv_sJUEO'] = pv_sJUEO

                    print("connected component analysis")
                    ind_entirely_uncert, ind_proportion_uncertain, ind_mean_uncert, ind_sizes = conn_comp_analysis(means, ent_maps, rater0, rater1)
                    mean_ind_entirely_uncert = torch.stack([torch.Tensor([torch.Tensor(tr).mean() for tr in indr]) for indr in ind_entirely_uncert])
                    mean_ind_proportion_uncertain = torch.stack([torch.Tensor([torch.Tensor(tr).mean() for tr in indr]) for indr in ind_proportion_uncertain])
                    mean_ind_mean_uncert = torch.stack([torch.Tensor(indr).mean() for indr in ind_entirely_uncert])

                    for ti, t in enumerate(uncertainty_thresholds):
                        overall_results[num_samples][f'mean_ind_entirely_uncert_t{t:.2f}'] = mean_ind_entirely_uncert[:,ti]
                        overall_results[num_samples][f'mean_ind_proportion_uncertain_t{t:.2f}'] = mean_ind_proportion_uncertain[:,ti]
                    overall_results[num_samples]['mean_ind_mean_uncert'] = mean_ind_mean_uncert
                    # overall_results[num_samples]['ind_sizes'] = ind_sizes

                    print("pixelwise analysis")
                    JTP, JFP, JFN, IR = pixelwise_metrics(means, ent_maps, rater0, rater1, xs3d_test)
                    edJTP, edJFP, edJFN, edIR = edge_deducted_pixelwise_metrics(means, ent_maps, rater0, rater1, xs3d_test)
                    pixelwise_and_cc_results[num_samples]['JTP'] = torch.cat(JTP)
                    pixelwise_and_cc_results[num_samples]['JFP'] = torch.cat(JFP)
                    pixelwise_and_cc_results[num_samples]['JFN'] = torch.cat(JFN)
                    pixelwise_and_cc_results[num_samples]['IR'] = torch.cat(IR)
                    pixelwise_and_cc_results[num_samples]['edJTP'] = torch.cat(edJTP) 
                    pixelwise_and_cc_results[num_samples]['edJFP'] = torch.cat(edJFP)
                    pixelwise_and_cc_results[num_samples]['edJFN'] = torch.cat(edJFN)
                    pixelwise_and_cc_results[num_samples]['edIR'] = torch.cat(edIR)
                    # np.savez("/home/s2208943/ipdis/results/pixel_wise_and_cc_inter_rater_stats/" + f"voxelwise_IRstats_{args.dataset}_{args.uncertainty_type}_cv{args.cv_split}_ns{num_samples}.npz", **pixelwise_and_cc_results[num_samples])
                    
                    print("volume difference distribution information")
                    vds_rater0, vds_rater1, vds_rater_mean, sample_vol_skew = vd_dist_and_skew(samples, rater0, rater1)
                    vds_rater0 = torch.Tensor(vds_rater0)
                    vds_rater1 = torch.Tensor(vds_rater1)
                    vds_rater_mean = torch.Tensor(vds_rater_mean)
                    for ns in range(num_samples):
                        overall_results[num_samples][f'vds_rater0_sample{ns}'] = vds_rater0[:,ns]
                        overall_results[num_samples][f'vds_rater1_sample{ns}'] = vds_rater1[:,ns]
                        overall_results[num_samples][f'vds_rater_mean_sample{ns}'] = vds_rater_mean[:,ns]
                    overall_results[num_samples]['sample_vol_skew'] = sample_vol_skew

                if num_samples in all_result_ns:
                    print("COLLECTING ALL VVC2 RESULTS")
                    ccv2_all_overall, ccv2_all_pixelwise_and_cc = connected_component_analysis_v2(xs3d_test, means, ent_maps, rater0, rater1, pv_region_masks, region='all')
                    print("COLLECTING DEEP VVC2 RESULTS")
                    ccv2_deep_overall, ccv2_deep_pixelwise_and_cc = connected_component_analysis_v2(xs3d_test, means, ent_maps, rater0, rater1, pv_region_masks, region='deep')
                    print("COLLECTING PV VVC2 RESULTS")
                    ccv2_pv_overall, ccv2_pv_pixelwise_and_cc = connected_component_analysis_v2(xs3d_test, means, ent_maps, rater0, rater1, pv_region_masks, region='pv')
                    
                    pixelwise_and_cc_results[num_samples].update(ccv2_all_pixelwise_and_cc)
                    pixelwise_and_cc_results[num_samples].update(ccv2_deep_pixelwise_and_cc)
                    pixelwise_and_cc_results[num_samples].update(ccv2_pv_pixelwise_and_cc)
                    np.savez("/home/s2208943/ipdis/results/pixel_wise_and_cc_inter_rater_stats/" + f"voxelwise_IRstats_{args.dataset}_{args.uncertainty_type}_cv{args.cv_split}_ns{num_samples}.npz", **pixelwise_and_cc_results[num_samples])
                    
                    overall_results[num_samples].update(ccv2_all_overall)
                    overall_results[num_samples].update(ccv2_deep_overall)
                    overall_results[num_samples].update(ccv2_pv_overall)
                    
                # best dice when sorting the sample for dice
                print("best dice and GED results sorted by dice")
                overall_results[num_samples]['GED_dice_sorted'] = []
                for r, rater in enumerate(raters):
                    best_dice = []
                    for i, s in tqdm(enumerate(samples)):                
                        y = rater[i].cuda()
                        s = reorder_samples_by_dice(s, y)
                        best_dice.append(fast_dice(s[-1].cuda().argmax(dim=1), y))
                        overall_results[num_samples][f'rater{r}_best_dice_dsorted_ss{num_samples}'] = best_dice

                        if r == 0:
                            # ged score by dice
                            overall_results[num_samples]['GED_dice_sorted'].append(individual_multirater_iou_GED(means[i], [r[i] for r in raters], s))
                    
                overall_results[num_samples]['mean_gt_vols'] = mean_gt_vols
                
                path = "/home/s2208943/ipdis/WMH_UQ_assessment/trustworthai/journal_run/interrater_experiments/results/"
                # for key in overall_results[num_samples].keys():
                    # print(key, len(overall_results[num_samples][key]))
                pd.DataFrame(overall_results[num_samples]).to_csv(path + f"inter_rater_{args.dataset}_{args.uncertainty_type}_cv{args.cv_split}_ns{num_samples}.csv")
                
            else:
                print("samples is None, breaking now")
                break

        except Exception as e:
            print(f"failed at {num_samples}")
            print(e)
            raise e
            break

if __name__ == '__main__':
    parser = construct_parser()
    args = parser.parse_args()
    main(args)
