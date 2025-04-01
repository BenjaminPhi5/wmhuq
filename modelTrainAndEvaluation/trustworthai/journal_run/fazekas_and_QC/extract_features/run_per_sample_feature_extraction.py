print("strawberry")
# trainer
from trustworthai.utils.fitting_and_inference.fitters.basic_lightning_fitter import StandardLitModelWrapper
from trustworthai.utils.fitting_and_inference.get_trainer import get_trainer

# data
from twaidata.torchdatasets_v2.mri_dataset_inram import MRISegmentation3DDataset
from twaidata.torchdatasets_v2.mri_dataset_from_file import MRISegmentationDatasetFromFile, ArrayMRISegmentationDatasetFromFile
from twaidata.mri_dataset_directory_parsers.MSS3_multirater import MSS3MultiRaterDataParser
from torch.utils.data import ConcatDataset

# packages
import math
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torchinfo import summary
from tqdm import tqdm
from collections import defaultdict
from natsort import natsorted
import torchmetrics
import argparse
import cc3d
from collections import defaultdict
from typing import Iterable

from trustworthai.journal_run.new_MIA_fazekas_and_QC.extract_features.utils import *
print("banana")

region_map = {
    0:'ring1',
    1:'ring2',
    2:'ring3',
    3:'ring4',
    4:'thalamus',
    5:'caudate',
    6:'thalamus+caudate',
    7:'above_thalamus_pv',
    8:'above_thalamus_deep',
}

def reorder_pred_samples(sample):
    sample = torch.from_numpy(sample)
    sample = sample.cuda()
    slice_volumes = (sample > 0.5).sum(dim=(-1, -2))
    slice_volume_orders = torch.sort(slice_volumes.T, dim=1)[1]
    
    # rearrange the samples into one...
    new_sample = torch.zeros(sample.shape).to(sample.device)
    for i, slice_volumes_orders in enumerate(slice_volume_orders):
        for j, sample_index in enumerate(slice_volumes_orders):
            new_sample[j][i] = sample[sample_index][i]
            
    return new_sample.cpu()

def load_samples_system(sample_maps_dir):
    sample_maps = {}
    for fID in tqdm(os.listdir(sample_maps_dir), position=0, leave=True, ncols=100):
        if "model_WMH_samples.npz" in fID:
            ID = fID.split("_model_WMH_samples")[0]
            sample_maps_data = np.load(os.path.join(sample_maps_dir,fID))
            samples = reorder_pred_samples(sample_maps_data['samples'])
            sample_maps[ID] = samples
            
    return sample_maps

def get_conn_comps(img):
    connectivity = 6 # only 4,8 (2D) and 26, 18, and 6 (3D) are allowed
    labels_out_seg = cc3d.connected_components(img.cpu().numpy(), connectivity=connectivity)
    return torch.from_numpy(labels_out_seg.astype(np.int16)).to(img.device)

def get_vent_rings(vent_dist_map):
    ring1 = vent_dist_map <= 5
    ring2 = (5 < vent_dist_map) * (vent_dist_map <= 10)
    ring3 = (10 < vent_dist_map) * (vent_dist_map <= 15)
    ring4 = vent_dist_map > 15
    
    return [ring1, ring2, ring3, ring4]


def get_extra_regions(rings, synthseg_out):
    """
    gets the thalamus and caudate regions, as well as the region above the highest point of the thalamus within the 10mm ring and the region above the highest point of the thalamus outside the 10mm ring.
    A potential problem here is that now I have overlapping features, where these overlap with ring3 and ring4....
    """
    
    # thalamus and caudate
    thalamus = (synthseg_out == 11) | (synthseg_out == 50)
    caudate = (synthseg_out == 10) | (synthseg_out == 49)
    
    thalamus_plus_caudate = thalamus + caudate

    wheres_thalamus_top = torch.where(thalamus)[0][-1]
    
    above_thalamus_pv_region = rings[2].clone()
    above_thalamus_pv_region[:wheres_thalamus_top] = 0
    above_thalamus_deep_region = torch.logical_not(rings[2].clone())
    above_thalamus_deep_region[:wheres_thalamus_top] = 0
    
    return [thalamus, caudate, thalamus_plus_caudate, above_thalamus_pv_region, above_thalamus_deep_region]


def calc_features(synthseg_out, vent_dist_map, img, thresh, binarize=False):
    ccs = get_conn_comps(img > thresh)
    rings = get_vent_rings(vent_dist_map)
    
    extra_regions = get_extra_regions(rings, synthseg_out)
    
    regions = torch.stack(rings + extra_regions)
    
    if binarize:
        img = img > thresh
    else:
        img = img * (img > thresh)
    
    ### overall sum
    wmhsum = img.sum().item()
    
    ### sums per ring
    wmh_region_sums = [
        (img * region).sum().item() for region in regions
    ]
    
    ### mean and std of cc sizes, per region and num ccs per region
    cc_sizes_per_region = [[] for _ in range(regions.shape[0])]
    ccs_in_each_region = []
    for i, region in enumerate(regions):
        ccs_region = ccs * region
        ccs_in_region = torch.unique(ccs_region)
        ccs_in_region = [cc for cc in ccs_in_region if cc != 0]
        ccs_in_each_region.append(ccs_in_region)
        
        # size of each cc's contribution to each region
        for cc in ccs_in_region:
            cc_sizes_per_region[i].append(((ccs_region == cc) * img).sum().item())
    
    # num_ccs_per_region
    num_ccs_per_region = [len(ccier) for ccier in ccs_in_each_region]
    
    # mean and std of cc sizes in each region
    cc_sizes_per_region = [torch.tensor(ccspr).type(torch.float32) for ccspr in cc_sizes_per_region]
    # print(cc_sizes_per_region)
    # print("after here")
    # print(cc_sizes_per_region[0].mean())
    # print("done")
    mean_cc_size_per_region = [ccspr.mean().item() for ccspr in cc_sizes_per_region]
    # for ccspr in cc_sizes_per_region:
    #     print(ccspr)
    #     print(ccspr.mean())
    std_cc_size_per_region = [ccspr.std(correction=1).item() if ccspr.shape[0] > 1 else 0 for ccspr in cc_sizes_per_region]
    
    ### and size of largest 'confluent' ccs
    # looking at region 1,2 and region 2,3 (2,3 is the 'confluence' region
    
    ccs_in_each_region = [set([ccid.item() for ccid in ccier]) for ccier in ccs_in_each_region] # convert to sets
    # print(ccs_in_each_region)
    
    ring_1_2_intersection = ccs_in_each_region[0] & ccs_in_each_region[1]
    ring_2_3_intersection = ccs_in_each_region[1] & ccs_in_each_region[2]
    
    def max_cc_size(cc_set):
        max_size = 0
        for cc in cc_set:
            cc_sum = ((ccs == cc) * img).sum().item()
            if cc_sum > max_size:
                max_size= cc_sum
        return max_size
    
    ring12_confluence = max_cc_size(ring_1_2_intersection)
    ring23_confluence = max_cc_size(ring_2_3_intersection)
    
    return {
        "wmhsum":wmhsum,
        "wmh_region_sums":wmh_region_sums,
        "region_cc_num":num_ccs_per_region,
        "mean_cc_region_sizes":mean_cc_size_per_region,
        "std_cc_region_sizes":std_cc_size_per_region,
        "region12_confluence":ring12_confluence,
        "region23_confluence":ring23_confluence,
    }

def calc_img_features(synthseg_out, vent_dist_map, img, never_binarize=False):
    thresh_feats = {}
    for threshold in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        if threshold == 0.1 or never_binarize:
            feats = calc_features(synthseg_out, vent_dist_map, img, threshold, binarize=False)
        else:
            feats = calc_features(synthseg_out, vent_dist_map, img, threshold, binarize=True)
        
        thresh_feats[threshold] = feats
    
    return thresh_feats

def get_features(all_samples, model_outs, IDs, vent_dists, synthseg_outs):
    IDs = np.array(IDs)
    
    pid_feats = {}
    
    for pid in tqdm(IDs, position=0, leave=True, ncols=100):
        scan_id = np.where(IDs == pid)[0].item()
        
        vent_dist = vent_dists[scan_id]
        synthseg_out = synthseg_outs[scan_id]
        if vent_dist is None:
            continue
        synthseg_out = torch.from_numpy(synthseg_out).cuda()
        vent_dist = torch.from_numpy(vent_dist).cuda()
        if pid in all_samples:
            samples = all_samples[pid].cuda()
        mean_pred = model_outs[pid][1].cuda()
        umap = model_outs[pid][0].cuda().clone() / -math.log(0.5)
        umap[mean_pred > 0.5] = 1 # fill in the uncertainty map where the prediction is
        
        if pid in all_samples:
            sample_feats = {f"sample_{i}":calc_img_features(synthseg_out, vent_dist, s) for (i, s) in enumerate(samples)}
        mean_feats = {f"mean":calc_img_features(synthseg_out, vent_dist, mean_pred)}
        umap_feats_binarized = {"umap_binarized":calc_img_features(synthseg_out, vent_dist, umap)}
        umap_feats_raw = {"umap_raw":calc_img_features(synthseg_out, vent_dist, umap, never_binarize=True)}
        
        if pid in all_samples:
            pid_feats[pid] = sample_feats
            pid_feats[pid].update(mean_feats)
        else:
            pid_feats[pid] = mean_feats
        pid_feats[pid].update(umap_feats_binarized)
        pid_feats[pid].update(umap_feats_raw)
    
    return pid_feats
        

def construct_parser():
    parser = argparse.ArgumentParser(description = "extract features for fazekas prediction model")
    
    parser.add_argument('--model_name', required=True)
    parser.add_argument('--dataset_name', required=True)
    
    return parser

def main(args):
    model_name = args.model_name
    
    ds_name = args.dataset_name
    
    if ds_name == "Ed_CVD":
        domains_ed = ["domainA", "domainB", "domainC", "domainD"]
        ds = ConcatDataset([
            MRISegmentation3DDataset("/home/s2208943/preprocessed_data/Ed_CVD/collated", no_labels=True, xy_only=False, domain_name=dn)
            for dn in domains_ed
        ])
        
        output_maps_dirs = [f"/home/s2208943/preprocessed_data/Ed_CVD/EdData_output_maps/{model_name}/"]
        out_folder_name = "/home/s2208943/preprocessed_data/Ed_CVD/EdData_feature_spreadsheets"
        synthseg_dirs = [f'/home/s2208943/preprocessed_data/Ed_CVD/{dn}/imgs/' for dn in domains_ed]
   
    elif ds_name == "ADNI300":
        ds = MRISegmentation3DDataset("/home/s2208943/preprocessed_data/ADNI300/collated", no_labels=True, xy_only=False)
        output_maps_dirs = [f"/home/s2208943/preprocessed_data/ADNI300/ADNI_300_output_maps/{model_name}/"]
        out_folder_name = '/home/s2208943/preprocessed_data/ADNI300/ADNI_300_feature_spreadsheets'
        synthseg_dirs = ['/home/s2208943/preprocessed_data/ADNI300/imgs/']
        
    elif ds_name == "Challenge":
        domains_chal = ["training_Singapore", "training_Utrecht", "training_Amsterdam_GE3T", "test_Amsterdam_GE1T5", "test_Amsterdam_Philips_VU_PETMR_01", "test_Utrecht", "test_Amsterdam_GE3T", "test_Singapore"]
    
        ds = ConcatDataset([
            MRISegmentation3DDataset("/home/s2208943/preprocessed_data/WMHChallenge_InterRaterData/collated", no_labels=True, xy_only=False, domain_name=dn)
            for dn in domains_chal
        ])
        
        output_maps_dirs = [f"/home/s2208943/preprocessed_data/WMHChallenge_InterRaterData/output_maps/training/{model_name}/", f"/home/s2208943/preprocessed_data/WMHChallenge_InterRaterData/output_maps/test/{model_name}/"] 
        out_folder_name = "/home/s2208943/preprocessed_data/WMHChallenge_InterRaterData/feature_spreadsheets"
        synthseg_dirs = [f'/home/s2208943/preprocessed_data/WMHChallenge_InterRaterData/{dn.split("_")[0]}/{"_".join(dn.split("_")[1:])}/imgs/' for dn in domains_chal]
        print(synthseg_dirs)
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
    
    else:
        raise ValueError("ds unknown")

    ### load base dataset
    
    
    IDs = [v[2] for v in ds]

    ### load model outputs
    print("LOADING MODEL OUTPUTS")
    output_maps = {}
    key_order = [] 
    for omd in output_maps_dirs:
        om, k = load_output_maps(omd)
        output_maps.update(om)
        key_order.extend(k)
        
    ### load synthseg outputs
    print("LOADING SYNTHSEG OUTPUTS")
    # added a whole load of extra logic for the challenge dataset. In future, I need to rework the preprocessing so that the Id's match how the rest of the datsets work
    synthseg_outs = []
    vent_dists = []
    failure_ids = []
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
                break
            except Exception as e:
                es.append(e)
                continue
        if synthseg is None:
            print("failed to load synthseg for : ", patient_id)
            # for e in es:
            #     print(e)
        synthseg_outs.append(synthseg)
        vent_dists.append(vent_d)
    
    ### load the samples
    all_samples = {}
    if model_name != "deterministic":
        for omd in output_maps_dirs:
            all_samples.update(load_samples_system(omd))
    
    ### run the analysis
    all_features = get_features(all_samples, output_maps, IDs, vent_dists, synthseg_outs)
   
    ### convert into a single dataframe
    df = defaultdict(lambda : [])

    
    for pid in all_features.keys():
        df['pID'].append(pid)
        pid_df = all_features[pid]
        for prediction_key in pid_df.keys():
            for thresh in pid_df[prediction_key].keys():
                thresh_df = pid_df[prediction_key][thresh]
                for feat in thresh_df.keys():
                    feat_value = thresh_df[feat]
                    if isinstance(feat_value, Iterable): # then it must be a region feature
                        for region in range(len(feat_value)):
                            df[f"{prediction_key}_t{thresh}_{feat}_{region_map[region]}"].append(thresh_df[feat][region])
                    else:
                        df[f"{prediction_key}_t{thresh}_{feat}"].append(thresh_df[feat])
    
    df = pd.DataFrame(df)
    IDs_map = {ID:"_".join(ID.split("_")[1:4]) for ID in IDs}
    reverse_IDs_map = {value:key for key, value in IDs_map.items()}
    df['ID'] = [IDs_map[pid] for pid in df['pID'].values]
    
    ### save the df
    df.to_csv(os.path.join(out_folder_name, f"{model_name}_per_sample_new_features_v2.csv"))
    
if __name__ == '__main__':
    parser = construct_parser()
    args = parser.parse_args()
    main(args)
