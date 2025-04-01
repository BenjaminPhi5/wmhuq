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
from twaidata.torchdatasets_v2.mri_dataset_inram import MRISegmentation3DDataset
from twaidata.torchdatasets_v2.mri_dataset_from_file import MRISegmentationDatasetFromFile, ArrayMRISegmentationDatasetFromFile
from twaidata.mri_dataset_directory_parsers.MSS3_multirater import MSS3MultiRaterDataParser
from trustworthai.utils.data_preprep.dataset_pipelines import load_data, ClinScoreDataRetriever

# evaluation code
from trustworthai.journal_run.evaluation.new_scripts.model_predictions import *
from trustworthai.journal_run.new_MIA_fazekas_and_QC.generating_model_outputs.utils import *
from trustworthai.journal_run.new_MIA_fazekas_and_QC.extract_features.utils import load_synthseg_data


import pdb
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
    
    # setup xent reweighting factor
    XENT_VOXEL_RESCALE = VOXELS_TO_WMH_RATIO - (1-args.empty_slice_retention) * (VOXELS_TO_WMH_RATIO - VOXELS_TO_WMH_RATIO_EXCLUDING_EMPTY_SLICES)

    XENT_WEIGHTING = XENT_VOXEL_RESCALE/2
    args.xent_reweighting = XENT_WEIGHTING
    
    ds_name = args.dataset
    model_name = args.model_name
    
    if ds_name == "Ed_CVD":
        domains_ed = ["domainA", "domainB", "domainC", "domainD"]
        
        # get the 3d axial slice dataloaders
        clin_retriever = ClinScoreDataRetriever(use_updated_scores=True)

        train_ds_clin, val_ds_clin, test_ds_clin = clin_retriever.load_clinscores_data(
                combine_all=False,
                test_proportion=args.test_split, 
                validation_proportion=args.val_split,
                seed=args.seed,
                cross_validate=args.cross_validate,
                cv_split=args.cv_split,
                cv_test_fold_smooth=args.cv_test_fold_smooth,
            )
        
        ds = test_ds_clin
        
        output_maps_dirs = [f"/home/s2208943/preprocessed_data/Ed_CVD/EdData_output_maps/{model_name}/"]
        out_folder_name = "/home/s2208943/preprocessed_data/Ed_CVD/EdData_feature_spreadsheets"
        synthseg_dirs = [f'/home/s2208943/preprocessed_data/Ed_CVD/{dn}/imgs/' for dn in domains_ed]
        model_result_folder = "/home/s2208943/preprocessed_data/Ed_CVD/EdData_feature_spreadsheets/"
        
    elif ds_name == "ADNI300":
        ds = MRISegmentation3DDataset("/home/s2208943/preprocessed_data/ADNI300/collated", no_labels=True, xy_only=False)
        output_maps_dirs = [f"/home/s2208943/preprocessed_data/ADNI300/ADNI_300_output_maps/{model_name}/"]
        out_folder_name = '/home/s2208943/preprocessed_data/ADNI300/ADNI_300_feature_spreadsheets'
        synthseg_dirs = ['/home/s2208943/preprocessed_data/ADNI300/imgs/']
        model_result_folder = "/home/s2208943/preprocessed_data/ADNI300/ADNI_300_feature_spreadsheets/"
        
    elif ds_name == "Challenge":
        domains_chal = ["training_Singapore", "training_Utrecht", "training_Amsterdam_GE3T", "test_Amsterdam_GE1T5", "test_Amsterdam_Philips_VU_PETMR_01", "test_Utrecht", "test_Amsterdam_GE3T", "test_Singapore"]
    
        ds = ConcatDataset([
            MRISegmentation3DDataset("/home/s2208943/preprocessed_data/WMHChallenge_InterRaterData/collated", no_labels=False, xy_only=False, domain_name=dn)
            for dn in domains_chal
        ])
        
        output_maps_dirs = [f"/home/s2208943/preprocessed_data/WMHChallenge_InterRaterData/output_maps/training/{model_name}/", f"/home/s2208943/preprocessed_data/WMHChallenge_InterRaterData/output_maps/test/{model_name}/"] 
        out_folder_name = "/home/s2208943/preprocessed_data/WMHChallenge_InterRaterData/feature_spreadsheets"
        synthseg_dirs = [f'/home/s2208943/preprocessed_data/WMHChallenge_InterRaterData/{dn.split("_")[0]}/{"_".join(dn.split("_")[1:])}/imgs/' for dn in domains_chal]
        model_result_folder = "/home/s2208943/preprocessed_data/WMHChallenge_InterRaterData/feature_spreadsheets/"
    elif ds_name == "MSS3":
        parser = MSS3MultiRaterDataParser(
            # paths on the cluster for the in house data
            "/home/s2208943/datasets/Inter_observer",
            "/home/s2208943/preprocessed_data/MSS3_InterRaterData"
        )

        ds = ArrayMRISegmentationDatasetFromFile(parser) 
        output_maps_dirs = [f"/home/s2208943/preprocessed_data/MSS3_InterRaterData/output_maps/{model_name}"]
        out_folder_name = "/home/s2208943/preprocessed_data/MSS3_InterRaterData/feature_spreadsheets/"
        synthseg_dirs = ['/home/s2208943/preprocessed_data/MSS3_InterRaterData/imgs/']
        model_result_folder = "/home/s2208943/preprocessed_data/MSS3_InterRaterData/feature_spreadsheets/"
    
    else:
        raise ValueError("ds unknown")
        
    if not os.path.exists(model_result_folder):
        os.makedirs(model_result_folder)
    
    IDs_all = [out[2] for out in ds]
    
    # load the model
    print("LOADING INITIAL MODEL")
    model_dir = os.path.join(args.ckpt_dir, args.model_name)  
    print("model dir: ", model_dir)
    model_raw, loss, val_loss = MODEL_LOADERS[args.model_type](args)
    model = load_best_checkpoint(model_raw, loss, model_dir, punet=args.model_type == "punet")
        
    IDs = IDs_all
    
    print("output file will be: ")
    if ds_name == "Ed_CVD":
        outputfile = os.path.join(model_result_folder, f"{args.uncertainty_type}_sample_div_and_metrics_cv{args.cv_split}.csv")
    else:
        outputfile = os.path.join(model_result_folder, f"{args.uncertainty_type}_sample_div_and_metrics.csv")
    
    print(outputfile)
        
    # get predictions
    print("generating predictions")
    means, samples, misc = get_means_and_samples(model_raw, ds, num_samples=10, model_func=MODEL_OUTPUT_GENERATORS[args.uncertainty_type], args=args)

    # get uncertainty maps
    print("calculating uncertainty maps")
    ent_maps = get_uncertainty_maps(means, samples, misc, args)

    # load synthseg information
    print("LOADING SYNTHSEG OUTPUTS")
    # added a whole load of extra logic for the challenge dataset. In future, I need to rework the preprocessing so that the Id's match how the rest of the datsets work
    vent_dists = []
    for data in tqdm(ds):
        synthseg = None
        vent_d = None
        flair = data[0][0]
        patient_id = data[2]
        # print(patient_id)
        es = []
        if ds_name == "Challenge":
            patient_id = patient_id.split("_")[-1]
            # print(patient_id)
        for folder in synthseg_dirs:
            if ds_name == "Challenge":
                if "_".join(patient_id.split("_")[1:-1]) not in folder:
                    continue
            try:
                synthseg, vent_d = load_synthseg_data(folder, patient_id, flair)
                # print("success")
                # print(flair.shape, vent_d.shape)
                break
            except Exception as e:
                es.append(e)
                continue
        if synthseg is None:
            print("failed to load synthseg for : ", patient_id)
            vent_dists.append(None)
        else:
            vent_dists.append(vent_d)
    
    # collect the dice, avd, f1 and recall data
    dices = []
    avds = []
    f1s = []
    recalls = []
    for i in tqdm(range(len(means)), position=0, leave=True):
        pred = means[i].argmax(dim=1)
        if ds_name != "ADNI300":
            if ds_name != "MSS3":
                target = ds[i][1][0].type(torch.int64)
            else:
                target = ds[i][1][-1].type(torch.int64)
                # try:
                #     target = ds[i][1][-2].type(torch.int64)
                # except:
                #     print("MSS3 SWITCH TO MASK 0 for individual: ", IDs[i])
                #     target = ds[i][1][0].type(torch.int64)

            dices.append(getDSC(target, pred))
            avds.append(getAVD(target.numpy(), pred.numpy()))
            recall, f1 = getLesionDetection(target.numpy(), pred.numpy())
            recalls.append(recall)
            f1s.append(f1)
            
    df = {
        "ID":IDs,
        "dice":dices,
        "f1":f1s,
        "avd":avds,
        "recall":recalls,
    } if ds_name != "ADNI300" else {"ID":IDs}
    
    ### collecting the sample div data
    if samples[0] != None:
        # rerrange the samples into sorted order
        samples = [reorder_samples(s) for s in samples]

        # get the sample diversity information
        results = defaultdict(lambda : [])
        for i, s in tqdm(enumerate(samples), position=0, leave=True, total=len(samples)):
            vmap = vent_dists[i]
            if vmap is not None:
                vmap = torch.from_numpy(vmap).cuda()
            else:
                results[f"{region}_sample_div_std"].append(np.nan)
                results[f"{region}_sample_div_IQR"].append(np.nan)
                results[f"{region}_sample_div_skew"].append(np.nan)
                results[f"{region}_sample_div_vd_std"].append(np.nan)
                results[f"{region}_sample_div_vd_IQR"].append(np.nan)
                results[f"{region}_sample_div_vd_skew"].append(np.nan)
                results[f"{region}_min_sample"].append(np.nan)
                results[f"{region}_max_sample"].append(np.nan)
                results[f"{region}_25th_p_sample"].append(np.nan)
                results[f"{region}_75th_p_sample"].append(np.nan)
                results[f"{region}_25th_p_vd"].append(np.nan)
                results[f"{region}_75th_p_vd"].append(np.nan)
                continue
            
            mean_full = means[i].cuda().argmax(dim=1)

            s_pred_full = s.cuda().argmax(dim=2)
            
            # print(ds[i][0].shape, mean_full.shape, s_pred_full.shape, vmap.shape)
            
            for region in ['deep', 'pv', 'all']:
                if region == 'deep':
                    mask = (vmap > 10)
                    # print(mask.shape, mean_full.shape, s_pred_full.shape)
                    mean = mean_full * mask
                    s_pred = s_pred_full * mask.unsqueeze(0)
                elif region == 'pv':
                    # print(mask.shape, mean.shape, s_pred.shape)
                    mask = (vmap <= 10)
                    mean = mean_full * mask
                    s_pred = s_pred_full * mask.unsqueeze(0)
                else:
                    mean = mean_full
                    s_pred = s_pred_full
                
                mean_vol = mean.sum(dim=(-3, -2, -1)).item()
            
                sorted_sample_volumes = s_pred.sum(dim=(-3,-2,-1)).type(torch.float32).cpu()
                ss_vds = ((sorted_sample_volumes - mean_vol) / mean_vol) * 100
                
                percentiles_samples = np.percentile(sorted_sample_volumes, [25, 75])
                percentiles_vds = np.percentile(ss_vds, [25, 75])

                results[f"{region}_sample_div_std"].append(sorted_sample_volumes.std().item())
                results[f"{region}_sample_div_IQR"].append(scipy.stats.iqr(sorted_sample_volumes))
                results[f"{region}_sample_div_skew"].append(scipy.stats.skew(sorted_sample_volumes))
                results[f"{region}_sample_div_vd_std"].append(ss_vds.std().item())
                results[f"{region}_sample_div_vd_IQR"].append(scipy.stats.iqr(ss_vds))
                results[f"{region}_sample_div_vd_skew"].append(scipy.stats.skew(ss_vds))
                results[f"{region}_min_sample"].append(sorted_sample_volumes.min().item())
                results[f"{region}_max_sample"].append(sorted_sample_volumes.max().item())
                results[f"{region}_25th_p_sample"].append(percentiles_samples[0])
                results[f"{region}_75th_p_sample"].append(percentiles_samples[1])
                results[f"{region}_25th_p_vd"].append(percentiles_vds[0])
                results[f"{region}_75th_p_vd"].append(percentiles_vds[1])
        
        # this is much much faster than df.update(results)
        for key in results:
            df[key] = results[key]
        
    for key in df:
        print(f"{key}: {len(df[key])}")
    df = pd.DataFrame(df)
    if ds_name == "Ed_CVD":
        df.to_csv(os.path.join(model_result_folder, f"{args.uncertainty_type}_sample_div_and_metrics_cv{args.cv_split}.csv"))
    else:
        df.to_csv(os.path.join(model_result_folder, f"{args.uncertainty_type}_sample_div_and_metrics.csv"))


if __name__ == '__main__':
    parser = construct_parser()
    args = parser.parse_args()
    main(args)
