"""
feature extraction used for the fazekas and QC tasks.
"""

import math
import os
import numpy as np
import pandas as pd
import torch
from collections import defaultdict
from natsort import natsorted
import torchmetrics
import cc3d
import scipy.stats


region_map = {
    0:'ring1',
    1:'ring2',
    2:'ring3',
    3:'ring4',
    4:'thalamus',
    5:'thalamus+caudate',
    6:'above_thalamus_pv',
    7:'above_thalamus_deep',
}

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
    
    return [thalamus, thalamus_plus_caudate, above_thalamus_pv_region, above_thalamus_deep_region]


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

    mean_cc_size_per_region = [ccspr.mean().item() for ccspr in cc_sizes_per_region]
    mean_cc_size_per_region = [mccspr if ccir > 0 else 0 for (mccspr, ccir) in zip(mean_cc_size_per_region, num_ccs_per_region)] # correct for nan values

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

def calc_img_features(synthseg_out, vent_dist_map, img):
    thresh_feats = {}
    for threshold in [0.1, 0.2]:
        feats = calc_features(synthseg_out, vent_dist_map, img, threshold, binarize=False)
        thresh_feats[threshold] = feats
    
    return thresh_feats


def get_sample_keys(df):
    sample_keys = []
    for key in df.columns:
        if key.startswith("sample_"):
            key_type = "_".join(key.split("_")[2:])
            if key_type not in sample_keys:
                sample_keys.append(key_type)
    
    print("----\nSAMPLE KEYS")
    print(sample_keys)
    return sample_keys

def extract_sample_values(df, key):
    key_per_sample = df[[f'sample_{s}_{key}' for s in range(10)]]
    print("----\nKEY PER SAMPLE")
    print(key_per_sample)
    return key_per_sample

def uncertainty_dfs(df, sample_keys):
    
    new_cols = {}
    
    for key in sample_keys:
        key_samples = extract_sample_values(df, key).values.astype(np.float32)
        key_mean = df[f'mean_{key}'].values.reshape(-1, 1).astype(np.float32)
        vd_key_samples = 100 * (key_samples - key_mean) / (key_mean + 1e-5)
        
        new_cols[f'std_{key}'] = np.std(key_samples, axis=1)
        new_cols[f'iqr_{key}'] = scipy.stats.iqr(key_samples, axis=1)
        
        new_cols[f'vd_std_{key}'] = np.std(vd_key_samples, axis=1)
        new_cols[f'vd_iqr_{key}'] = scipy.stats.iqr(vd_key_samples, axis=1)
    
    new_cols = pd.DataFrame(new_cols)
    df = pd.concat([df, new_cols], axis=1)
    return df


def extract_features(synthseg_out, vent_dist, samples, mean_pred, umap):
    print("updated version")
    sample_feats = {f"sample_{i}":calc_img_features(synthseg_out, vent_dist, s) for (i, s) in enumerate(samples)}
    mean_feats = {f"mean":calc_img_features(synthseg_out, vent_dist, mean_pred)}
    umap_feats_raw = {"umap_raw":calc_img_features(synthseg_out, vent_dist, umap)}
    # umap2 = umap.clone()
    umap3 = umap.clone()
    # umap2[mean_pred >= 0.5] = 1 # fill in the uncertainty map where the prediction is
    # umap_feats_filled = {"umap_filled": calc_img_features(synthseg_out, vent_dist, umap2)}
    umap3[mean_pred < 0.5] = 0
    umap_feats_atseg = {"umap_atseg": calc_img_features(synthseg_out, vent_dist, umap3)}
    
    feats = sample_feats
    feats.update(mean_feats)
    feats.update(umap_feats_raw)
    # feats.update(umap_feats_filled)
    feats.update(umap_feats_atseg)
    
    df = {}
    for inp_key in feats.keys():
        for t_key in feats[inp_key].keys():
            for region_key in feats[inp_key][t_key].keys():
                if isinstance(feats[inp_key][t_key][region_key], list):
                    regions_list = feats[inp_key][t_key][region_key]
                    for index, subregion_key in region_map.items():
                        df[f'{inp_key}_t{t_key}_{region_key}_{subregion_key}'] = [regions_list[index]]
                else:
                    df[f'{inp_key}_t{t_key}_{region_key}'] = [feats[inp_key][t_key][region_key]]
    
    feats = pd.DataFrame(df)
    print("\n\n\nKNOWN COLUMNS")
    print(feats.columns)
    print("#########\n##############\n\n##############")
    feats = uncertainty_dfs(feats, get_sample_keys(feats))
    
    return feats
