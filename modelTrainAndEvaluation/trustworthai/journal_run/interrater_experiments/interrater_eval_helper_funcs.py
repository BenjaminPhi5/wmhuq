import torch
import torch.nn.functional as F

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


from scipy.ndimage import distance_transform_edt
import SimpleITK as sitk
from twaidata.MRI_preprep.resample import get_resampled_img
import cc3d

uncertainty_thresholds = torch.arange(0, 0.7, 0.01)

def UIRO(pred, thresholded_umap, seg1, seg2):
    IR = (seg1 != seg2)
    error = (seg1 == seg2) * (seg1 != pred)
    thresholded_umap[error] = 0
    return fast_dice(thresholded_umap, IR)


def JUEO(pred, thresholded_umap, seg1, seg2):
    IR = (seg1 != seg2)
    error = (seg1 == seg2) * (seg1 != pred)
    thresholded_umap[IR] = 0
    return fast_dice(thresholded_umap, error)

def per_rater_UEO(pred, thresholded_umap, seg1, seg2):
    error1 = (seg1 != pred)
    error2 = (seg2 != pred)
    
    return fast_dice(thresholded_umap, error1), fast_dice(thresholded_umap, error2)

def per_threshold_ueos(means, ent_maps, rater0, rater1, xs3d_test):
    uiro_curves = []
    jueo_curves = []
    for i in tqdm(range(len(xs3d_test))):
        uiro = []
        jueo = []
        m = means[i].argmax(dim=1).cuda()
        for t in uncertainty_thresholds:
            e = ent_maps[i] > t
            e = e.cuda()
            uiro.append(UIRO(m, e.clone(), rater0[i].cuda(), rater1[i].cuda()))
            jueo.append(JUEO(m, e.clone(), rater0[i].cuda(), rater1[i].cuda()))
        uiro_curves.append(uiro)
        jueo_curves.append(jueo)
        
    return uiro_curves, jueo_curves

def get_2rater_rmse(pred, y0, y1, p=0.1):
    label = torch.zeros(pred.shape, device='cuda')
    label[:,0] = (y0 == 0) & (y1 == 0)
    label[:,1] = (y0 == 1) & (y1 == 1)
    diff = (y0 != y1)
    label[:,0][diff] = 0.5
    label[:, 1][diff] = 0.5
    
    locs = pred[:,1] > p
    # print(pred.shape)
    
    pred = pred.moveaxis(1, -1)[locs]
    label = label.moveaxis(1, -1)[locs]
    
    rmse = ((pred - label).square().sum(dim=1) / pred.shape[1]).mean().sqrt()

    return rmse.item()

def get_IR_rmse(pred, y0, y1, p=0.1):
    label = torch.zeros(pred.shape, device='cuda')
    label[:,0] = (y0 == 0) & (y1 == 0)
    label[:,1] = (y0 == 1) & (y1 == 1)
    diff = (y0 != y1)
    label[:,0][diff] = 0.5
    label[:, 1][diff] = 0.5
    
    locs = diff
    # print(pred.shape)
    
    pred = pred.moveaxis(1, -1)[locs]
    label = label.moveaxis(1, -1)[locs]
    
    rmse = ((pred - label).square().sum(dim=1) / pred.shape[1]).mean().sqrt()

    return rmse.item()

def dilate(tensor, kernel_size=3, iterations=1):
    """
    Dilate a 3D binary tensor using a cubic kernel.
    
    Parameters:
    - tensor: A 3D binary tensor of shape (C, H, W, D) where C is the channel (1 for binary images).
    - kernel_size: Size of the cubic kernel for dilation.
    - iterations: Number of times dilation is applied.
    
    Returns:
    - Dilated tensor.
    """
    padding = kernel_size // 2
    kernel = torch.ones((1, 1, kernel_size, kernel_size, kernel_size), device=tensor.device)
    for _ in range(iterations):
        tensor = F.conv3d(tensor, kernel, padding=padding, groups=1)
    return torch.clamp(tensor, 0, 1)

def erode(tensor, kernel_size=3, iterations=1):
    """
    Erode a 3D binary tensor using a cubic kernel.
    
    Parameters:
    - tensor: A 3D binary tensor of shape (C, H, W, D).
    - kernel_size: Size of the cubic kernel for erosion.
    - iterations: Number of times erosion is applied.
    
    Returns:
    - Eroded tensor.
    """
    padding = kernel_size // 2
    kernel = torch.ones((1, 1, kernel_size, kernel_size, kernel_size), device=tensor.device)
    for _ in range(iterations):
        tensor = F.conv3d(1 - tensor, kernel, padding=padding, groups=1)
    return 1 - torch.clamp(tensor, 0, 1)

def find_edges(tensor, kernel_size=3):
    """
    Find the inner and outer edges of a segmentation in a 3D binary tensor.
    
    Parameters:
    - tensor: A 3D binary tensor of shape (H, W, D).
    - kernel_size: Size of the cubic kernel for dilation and erosion.
    
    Returns:
    - inner_edges: The inner edges of the segmentation.
    - outer_edges: The outer edges of the segmentation.
    """
    tensor = tensor.unsqueeze(0)
    dilated = dilate(tensor, kernel_size)
    eroded = erode(tensor, kernel_size)
    outer_edges = dilated - tensor
    inner_edges = tensor - eroded
    
    outer_edges = outer_edges.squeeze()
    inner_edges = inner_edges.squeeze()
    
    return inner_edges, outer_edges


def get_rmse_stats(means, rater0, rater1):
    rmses = []
    IR_rmses = []
    p = 0.1
    for i in tqdm(range(len(means))):
        y0 = rater0[i]
        y1 = rater1[i]
        m = means[i].cuda()
        m = m.softmax(dim=1)
        rmse = get_2rater_rmse(m, y0, y1, p)
        ir_rmse = get_IR_rmse(m, y0, y1, p)

        rmses.append(rmse)
        IR_rmses.append(ir_rmse)
        
    return rmses, IR_rmses
    
def edge_deducted_UIRO(pred, thresholded_umap, seg1, seg2):
    pred = pred.type(torch.float32).cuda()
    inner_edge, outer_edge = find_edges(pred)#.unsqueeze(0))
    seg1 = seg1.cuda()
    seg2 = seg2.cuda()
    
    IR = (seg1 != seg2)
    error = (seg1 == seg2) * (seg1 != pred)
    thresholded_umap[error] = 0
    
    non_edge_locs = ((1-inner_edge) * (1 - outer_edge)) == 1
    thresholded_umap *= non_edge_locs
    IR *= non_edge_locs
    
    return fast_dice(thresholded_umap, IR)


def edge_deducted_JUEO(pred, thresholded_umap, seg1, seg2):
    pred = pred.type(torch.float32).cuda()
    inner_edge, outer_edge = find_edges(pred)#.unsqueeze(0))
    seg1 = seg1.cuda()
    seg2 = seg2.cuda()
    
    IR = (seg1 != seg2)
    error = (seg1 == seg2) * (seg1 != pred)
    thresholded_umap[IR] = 0
    
    non_edge_locs = ((1-inner_edge) * (1 - outer_edge)) == 1
    thresholded_umap *= non_edge_locs
    error *= non_edge_locs
    
    return fast_dice(thresholded_umap, error)

def per_threshold_edge_deducted_ueos(means, ent_maps, rater0, rater1, xs3d_test):
    uiro_curves = []
    jueo_curves = []
    for i in tqdm(range(len(xs3d_test))):
        uiro = []
        jueo = []
        m = means[i].argmax(dim=1).cuda()
        for t in uncertainty_thresholds:
            e = ent_maps[i] > t
            e = e.cuda()
            uiro.append(edge_deducted_UIRO(m, e.clone(), rater0[i].cuda(), rater1[i].cuda()))
            jueo.append(edge_deducted_JUEO(m, e.clone(), rater0[i].cuda(), rater1[i].cuda()))
        uiro_curves.append(uiro)
        jueo_curves.append(jueo)
        
    return uiro_curves, jueo_curves

def soft_dice(pred, target):
    
    numerator = 2 * (pred * target).sum()
    denominator = (target**2).sum() + (pred**2).sum()
    
    return (numerator / denominator).item()


def soft_edge_deducted_UIRO(pred, umap, seg1, seg2):
    pred = pred.type(torch.float32).cuda()
    inner_edge, outer_edge = find_edges(pred)#.unsqueeze(0))
    seg1 = seg1.cuda()
    seg2 = seg2.cuda()
    
    IR = (seg1 != seg2)
    error = (seg1 == seg2) * (seg1 != pred)
    umap[error] = 0
    
    non_edge_locs = ((1-inner_edge) * (1 - outer_edge)) == 1
    umap *= non_edge_locs
    IR *= non_edge_locs
    
    return soft_dice(umap, IR)


def soft_edge_deducted_JUEO(pred, umap, seg1, seg2):
    pred = pred.type(torch.float32).cuda()
    inner_edge, outer_edge = find_edges(pred)#.unsqueeze(0))
    seg1 = seg1.cuda()
    seg2 = seg2.cuda()
    
    IR = (seg1 != seg2)
    error = (seg1 == seg2) * (seg1 != pred)
    umap[IR] = 0
    
    non_edge_locs = ((1-inner_edge) * (1 - outer_edge)) == 1
    umap *= non_edge_locs
    error *= non_edge_locs
    
    return soft_dice(umap, error)

def soft_UIRO(pred, umap, seg1, seg2):
    IR = (seg1 != seg2)
    error = (seg1 == seg2) * (seg1 != pred)
    umap[error] = 0
    return soft_dice(umap, IR)


def soft_JUEO(pred, umap, seg1, seg2):
    IR = (seg1 != seg2)
    error = (seg1 == seg2) * (seg1 != pred)
    umap[IR] = 0
    return soft_dice(umap, error)

def soft_per_rater_UEO(pred, umap, seg1, seg2):
    error1 = (seg1 != pred)
    error2 = (seg2 != pred)
    
    return soft_dice(umap, error1), soft_dice(umap, error2)

def soft_ueo_metrics(means, ent_maps, rater0, rater1, xs3d_test, pv_region_masks):
    sUIRO = []
    sJUEO = []
    deep_sUIRO = []
    deep_sJUEO = []
    pv_sUIRO = []
    pv_sJUEO = []
    sUEO_r1 = []
    sUEO_r2 = []
    s_ed_UIRO = []
    s_ed_JUEO = []
    for i in tqdm(range(len(xs3d_test))):
        m = means[i].argmax(dim=1).cuda()
        e = ent_maps[i].cuda()
        y0, y1 = rater0[i].cuda(), rater1[i].cuda()
        pv_region = pv_region_masks[i].cuda()
        deep_region = 1 - pv_region
        
        sUIRO.append(soft_UIRO(m, e.clone(), y0, y1))
        
        sJUEO.append(soft_JUEO(m, e.clone(), y0, y1))
        
        s_ed_UIRO.append(soft_edge_deducted_UIRO(m, e.clone(), y0, y1))
        s_ed_JUEO.append(soft_edge_deducted_JUEO(m, e.clone(), y0, y1))
        
        sueo1, sueo2 = soft_per_rater_UEO(m, e, y0, y1)
        sUEO_r1.append(sueo1)
        sUEO_r2.append(sueo2)
        
        if m.shape[0] != pv_region.shape[0] or m.shape[1] != pv_region.shape[1] or m.shape[2] != pv_region.shape[2]:
                continue
        deep_sUIRO.append(soft_UIRO(m * deep_region, e.clone() * deep_region, y0 * deep_region, y1 * deep_region))
        pv_sUIRO.append(soft_UIRO(m * pv_region, e.clone() * pv_region, y0 * pv_region, y1 * pv_region))
        deep_sJUEO.append(soft_JUEO(m * deep_region, e.clone() * deep_region, y0 * deep_region, y1 * deep_region))
        pv_sJUEO.append(soft_JUEO(m * pv_region, e.clone() * pv_region, y0 * pv_region, y1 * pv_region))
        
    return sUIRO, sJUEO, sUEO_r1, sUEO_r2, s_ed_UIRO, s_ed_JUEO, deep_sUIRO, deep_sJUEO, pv_sUIRO, pv_sJUEO

def conn_comp_analysis(means, ent_maps, rater0, rater1):
    
    ind_entirely_uncert = []
    ind_proportion_uncertain = []
    ind_mean_uncert = []
    ind_sizes = []

    for i in tqdm(range(len(means))):
        pred = means[i].cuda().argmax(dim=1)
        e = ent_maps[i].cuda()
        y0 = rater0[i]
        y1 = rater1[i]

        disagreement = (y0 != y1)

        ccs = cc3d.connected_components(disagreement.type(torch.int32).numpy(), connectivity=26) # 26-connected
        ccs = torch.from_numpy(ccs.astype(np.float32)).cuda()

        entirely_uncertain = [[] for _ in range(len(uncertainty_thresholds))]
        proportion_uncertain = [[] for _ in range(len(uncertainty_thresholds))]
        mean_uncert = []
        sizes = []
        
        if len(ccs.unique()) != 1:
            for cc_id in ccs.unique():
                if cc_id == 0:
                    continue
                cc = ccs == cc_id
                size = cc.sum().item()
                sizes.append(size)
                mean_uncert.append(e[cc].mean().item())

                for j, t in enumerate(uncertainty_thresholds):
                    et = e > t
                    uncert_cc_sum = (cc * et).sum().item()
                    proportion_uncertain[j].append(uncert_cc_sum / size)
                    entirely_uncertain[j].append(uncert_cc_sum == size)

        ind_entirely_uncert.append(entirely_uncertain)
        ind_proportion_uncertain.append(proportion_uncertain)
        ind_mean_uncert.append(mean_uncert)
        ind_sizes.append(sizes)
        
    return ind_entirely_uncert, ind_proportion_uncertain, ind_mean_uncert, ind_sizes

def pixelwise_metrics(means, ent_maps, rater0, rater1, xs3d_test):
    JTP = []
    JFP = []
    JFN = []
    IR = []
    for i in tqdm(range(len(xs3d_test))):
        m = means[i].argmax(dim=1).cuda().type(torch.long)
        e = ent_maps[i].cuda()
        y0, y1 = rater0[i].cuda().type(torch.long), rater1[i].cuda().type(torch.long)
        
        # flatten for indexing
        e = e.view(-1)
        m = m.view(-1)
        y0 = y0.view(-1)
        y1 = y1.view(-1)
        
        joint = (y0 == y1)
        ir = (y0 != y1)
        
        JTP.append(e[(joint * y0 * m)==1].cpu())
        JFP.append(e[(joint * (1 - y0) * m)==1].cpu())
        JFN.append(e[(joint * y0 * (1 - m))==1].cpu())
        IR.append(e[ir].cpu())
        
    return JTP, JFP, JFN, IR

def edge_deducted_pixelwise_metrics(means, ent_maps, rater0, rater1, xs3d_test):
    JTP = []
    JFP = []
    JFN = []
    IR = []
    for i in tqdm(range(len(xs3d_test))):
        m = means[i].argmax(dim=1).cuda().type(torch.long)
        inner_edge, outer_edge = find_edges(m.type(torch.float32))
        e = ent_maps[i].cuda()
        y0, y1 = rater0[i].cuda().type(torch.long), rater1[i].cuda().type(torch.long)
        
        # flatten for indexing
        e = e.view(-1)
        m = m.view(-1)
        y0 = y0.view(-1)
        y1 = y1.view(-1)
        
        # non edge area
        non_edge = (1-inner_edge) * (1-outer_edge)
        non_edge = non_edge.view(-1)
        
        joint = (y0 == y1) * non_edge
        ir = (y0 != y1) * (non_edge == 1)
        
        JTP.append(e[(joint * y0 * m)==1].cpu())
        JFP.append(e[(joint * (1 - y0) * m)==1].cpu())
        JFN.append(e[(joint * y0 * (1 - m))==1].cpu())
        IR.append(e[ir].cpu())
        
    return JTP, JFP, JFN, IR

# make sure that we have the volume difference per individual
# make sure that this collection is put by the samples that are collected by volume!!!!!
def vd_dist_and_skew(samples, rater0, rater1):
    vds_rater0 = []
    vds_rater1 = []
    vds_rater_mean = []
    sample_vol_skew = []
    for i, s in tqdm(enumerate(samples), total=len(samples)):
        y0 = rater0[i].cuda().sum().item()
        y1 = rater1[i].cuda().sum().item()
        y_mean = ( y0 + y1 ) / 2
        s = s.cuda().argmax(dim=2)

        vds_rater0.append([(((sj.sum() - y0) / y0) * 100).item() for sj in s])
        vds_rater1.append([(((sj.sum() - y1) / y1) * 100).item() for sj in s])
        vds_mean = [(((sj.sum() - y_mean) / y_mean) * 100).item() for sj in s]
        vds_rater_mean.append(vds_mean)
        sample_vol_skew.append(scipy.stats.skew(np.array(vds_mean), bias=True))
    
    return vds_rater0, vds_rater1, vds_rater_mean, sample_vol_skew

def fast_iou(pred, target):
    p1 = (pred == 1)
    t1 = (target == 1)
    intersection = (pred == 1) & (target == 1)
    numerator = intersection.sum()
    denominator = p1.sum() + t1.sum() - numerator
    return (numerator/(denominator + 1e-30)).item()

def individual_multirater_iou_GED(mean, rater_ys, sample):
    ged = 0
    ys = [r for r in rater_ys]
    ss = sample.cuda().argmax(dim=2)
    num_samples = ss.shape[0]

    dists_ab = 0
    count_ab = 0
    for s in ss:
        for y in ys:
            pred = s#.argmax(dim=1)
            dists_ab += (1 - fast_iou(pred, y.cuda()))
            # print(dists_ab)
            # print(s.shape)
            count_ab += 1

    dists_ab /= count_ab # num_samples # count should be num_samples * num_raters for consistent number of raters but Ive just done this count for now.
    dists_ab *= 2

    dists_aa = 0
    count_aa = 0
    for j, y1 in enumerate(ys):
        for k, y2 in enumerate(ys):
            if j == k:
                continue
            dists_aa += (1 - fast_iou(y1.cuda(), y2.cuda()))
            count_aa += 1

    dists_aa /= count_aa

    dists_bb = 0
    for j, s1 in enumerate(ss):
        for k, s2 in enumerate(ss):
            if j == k:
                continue
            dists_bb += (1 - fast_iou(s1, s2))

    dists_bb /= (num_samples * (num_samples - 1))

    ged = dists_ab - dists_aa - dists_bb
        
    return ged

def multirater_iou_GED(means, rater_ys, samples):
    geds = []
    
    for i in tqdm(range(len(means)), position=0, leave=True):
        ys = [r[i] for r in rater_ys]
        ss = samples[i].cuda().argmax(dim=2)
        num_samples = ss.shape[0]
        
        dists_ab = 0
        count_ab = 0
        for s in ss:
            for y in ys:
                pred = s#.argmax(dim=1)
                dists_ab += (1 - fast_iou(pred, y.cuda()))
                # print(dists_ab)
                # print(s.shape)
                count_ab += 1
        
        dists_ab /= count_ab # num_samples # count should be num_samples * num_raters for consistent number of raters but Ive just done this count for now.
        dists_ab *= 2
        
        dists_aa = 0
        count_aa = 0
        for j, y1 in enumerate(ys):
            for k, y2 in enumerate(ys):
                if j == k:
                    continue
                dists_aa += (1 - fast_iou(y1.cuda(), y2.cuda()))
                count_aa += 1
        
        dists_aa /= count_aa
        
        dists_bb = 0
        for j, s1 in enumerate(ss):
            for k, s2 in enumerate(ss):
                if j == k:
                    continue
                dists_bb += (1 - fast_iou(s1, s2))
        
        dists_bb /= (num_samples * (num_samples - 1))
        
        ged = dists_ab - dists_aa - dists_bb
        if not np.isnan(ged):
            geds.append(ged)
        #break
        
    return torch.Tensor(geds)

def find_non_overlapping_ccs(ccs_y0, y1):
    no_overlap_ccs = []
    ccs_size = []
    no_overlap_ccs_img = torch.zeros(y1.shape, device=y1.device, dtype=y1.dtype)
    
    for cc_id in ccs_y0.unique():
        if cc_id == 0:
            continue
        cc = ccs_y0 == cc_id
        if (cc * y1).sum() == 0:
            no_overlap_ccs.append(int(cc_id.item()))
            ccs_size.append(cc.sum().item())
            no_overlap_ccs_img += cc
    
    return no_overlap_ccs, ccs_size, no_overlap_ccs_img

def remove_ccids_with_0IOU(ccs_y0, ccs_y1, y0, y1):
    y0_ccids = ccs_y0.unique().type(torch.int32).tolist()
    y1_ccids = ccs_y1.unique().type(torch.int32).tolist()
    y0_ccids.remove(0)
    y1_ccids.remove(0)
    zero_overlap_y0 = []
    zero_overlap_y1 = []
    for cc_id in y0_ccids:
        cc = ccs_y0 == cc_id
        if (cc * y1).sum() == 0:
            zero_overlap_y0.append(cc_id)
    for cc_id in y1_ccids:
        if cc_id == 0:
            continue
        cc = ccs_y1 == cc_id
        if (cc * y0).sum() == 0:
            zero_overlap_y1.append(cc_id)
    for cc_id in zero_overlap_y0:
        y0_ccids.remove(cc_id)
    for cc_id in zero_overlap_y1:
        y1_ccids.remove(cc_id)
        
    return y0_ccids, y1_ccids

def find_ious_between_raters(ccs_y0, ccs_y1, y0, y1):
    
    # first narrow down the list to all the connected components that do not have zero overlap
    y0_ccids, y1_ccids = remove_ccids_with_0IOU(ccs_y0, ccs_y1, y0, y1)
    
    # loop through ccs to find the one with the highest iou
    cc_ious = []
    ccs_y1_match = []
    sizes = []
    match_sizes = []
    for cc_id0 in y0_ccids:
        cc0 = ccs_y0 == cc_id0
        cc0_sum = cc0.sum().item()
        best_iou = 0
        best_iou_cc_id = None
        match_size = None
        for cc_id1 in y1_ccids:
            cc1 = ccs_y1 == cc_id1
            iou = (cc0 & cc1).sum() / ((cc1 + cc0) > 0).sum()
            iou = iou.item()
            if iou >= best_iou:
                best_iou = iou
                best_iou_cc_id = cc_id1
                match_size = cc1.sum().item()
        # print(best_iou_cc_id)
        # y1_ccids.remove(best_iou_cc_id)
        cc_ious.append(best_iou)
        ccs_y1_match.append(best_iou_cc_id)
        sizes.append(cc0_sum)
        match_sizes.append(match_size)
        
    return cc_ious, y0_ccids, ccs_y1_match, sizes, match_sizes

def create_overlap_image(y0_cc_ids, y1_cc_ids, y0_ccs, y1_ccs):
    overlap_ccs_img = torch.zeros(y1_ccs.shape, device=y1_ccs.device, dtype=y1_ccs.dtype)
    for cc_id in y0_cc_ids:
        cc = y0_ccs == cc_id
        overlap_ccs_img[cc] = 1
    if y1_cc_ids is not None:
        for cc_id in y1_cc_ids:
            cc = y1_ccs == cc_id
            overlap_ccs_img[cc] = 1
        
    return overlap_ccs_img

def get_distance_map(img, rescale_ratio=3):
    
    # resample to 1x1x1 space
    device = img.device
    img = img.type(torch.float32)
    img = torch.nn.functional.interpolate(img.unsqueeze(0).unsqueeze(0), size=None, scale_factor=(rescale_ratio, 1, 1), mode='nearest', align_corners=None, recompute_scale_factor=None, antialias=False)
    img = img.squeeze().cpu().numpy()
    
    # compute distance map
    distance_map = distance_transform_edt(1 - img)
    
    # resample back to original space
    distance_map = torch.from_numpy(distance_map).to(device)
    distance_map = torch.nn.functional.interpolate(distance_map.unsqueeze(0).unsqueeze(0), size=None, scale_factor=(1/rescale_ratio, 1, 1), mode='trilinear', align_corners=None, recompute_scale_factor=None, antialias=False)
    distance_map = distance_map.squeeze()
    
    return distance_map

def IOU_region_UIRO(ent, rater_diff, no_overlap_distance_map, low_overlap_distance_map, high_overlap_distance_map):
        uiro_no_overlap = []
        uiro_low_overlap = []
        uiro_high_overlap = []
        
        d = 5
        no_exclusion_region = (no_overlap_distance_map < d) * (low_overlap_distance_map > d) * (high_overlap_distance_map > d)
        low_exclusion_region = (no_overlap_distance_map > d) * (low_overlap_distance_map < d) * (high_overlap_distance_map > d)
        high_exclusion_region = (no_overlap_distance_map > d) * (low_overlap_distance_map > d) * (high_overlap_distance_map < d)
        
        for t in uncertainty_thresholds:
            e = ent > t
            uiro_no_overlap.append(fast_dice(e * no_exclusion_region, rater_diff * no_exclusion_region))
            uiro_low_overlap.append(fast_dice(e * low_exclusion_region, rater_diff * low_exclusion_region))
            uiro_high_overlap.append(fast_dice(e * high_exclusion_region, rater_diff * high_exclusion_region))
        
        
        return uiro_no_overlap, uiro_low_overlap, uiro_high_overlap

def connected_component_analysis_v2(xs3d_test, means, ent_maps, rater0, rater1, pv_region_masks, region='all'):
    all_sizes = []
    all_ious = []
    all_match_sizes = []
    IOU0_ccs_sizes = []

    ent_no_overlap_all, ent_low_overlap_all, ent_high_overlap_all, ent_exact_overlap_all = [], [], [], []

    none_prop_uncert_all = [[] for _ in range(len(uncertainty_thresholds))]
    low_prop_uncert_all = [[] for _ in range(len(uncertainty_thresholds))]
    high_prop_uncert_all = [[] for _ in range(len(uncertainty_thresholds))]
    exact_prop_uncert_all = [[] for _ in range(len(uncertainty_thresholds))]
    non_useful_ccs_all = [[] for _ in range(len(uncertainty_thresholds))]

    num_FP_uncertainty_ccs_all = []
    mean_size_FP_uncertainty_ccs_all = []

    none_cc_mean = []
    low_cc_mean = []
    high_cc_mean = []
    exact_cc_mean = []

    uiro_no_overlap_all = []
    uiro_low_overlap_all = []
    uiro_high_overlap_all = []

    for i in tqdm(range(len(means))):
        x = xs3d_test[i].cuda()
        pred = means[i].cuda().argmax(dim=1)
        e = ent_maps[i].cuda()
        y0 = rater0[i]
        y1 = rater1[i]
        if region != "all":
            pv_region = pv_region_masks[i].cuda()
            if x.shape[1] != pv_region.shape[0] or x.shape[2] != pv_region.shape[1] or x.shape[3] != pv_region.shape[2]:
                continue
        if region == "pv":
            y0 = y0.cuda() * pv_region
            y1 = y1.cuda() * pv_region
            e = e * pv_region
            pred = pred * pv_region
        elif region == "deep":
            deep_region = 1 - pv_region
            y0 = y0.cuda() * deep_region
            y1 = y1.cuda() * deep_region
            e = e * deep_region
            pred = pred * deep_region

        ccs_y0 = cc3d.connected_components(y0.cpu().type(torch.int32).numpy(), connectivity=26) # 26-connected
        ccs_y0 = torch.from_numpy(ccs_y0.astype(np.float32)).cuda()

        ccs_y1 = cc3d.connected_components(y1.cpu().type(torch.int32).numpy(), connectivity=26) # 26-connected
        ccs_y1 = torch.from_numpy(ccs_y1.astype(np.float32)).cuda()

        y0 = y0.cuda()
        y1 = y1.cuda()

        diff_image = (y0 + y1) == 1
        any_prediction_image = (pred + y0 + y1) > 0 # anywhere where either the model or the raters predict


        # find type 1: connected components in either map with no overlap
        no_overlap_ccs_y0, no_overlap_ccs_size_y0, no_overlap_ccs_img_y0 = find_non_overlapping_ccs(ccs_y0, y1)
        no_overlap_ccs_y1, no_overlap_ccs_size_y1, no_overlap_ccs_img_y1 = find_non_overlapping_ccs(ccs_y1, y0)
        IOU0_ccs_sizes.extend(no_overlap_ccs_size_y0 + no_overlap_ccs_size_y1)
        combined_no_overlap_image = (no_overlap_ccs_img_y0 + no_overlap_ccs_img_y1).type(torch.int)

        # get information on the IOU of connected components between rater 0 and rater 1 where the IOU > 0
        ious, y0_ccids, y1_ccid_matches, sizes, match_sizes = find_ious_between_raters(ccs_y0, ccs_y1, y0, y1)
        all_ious.append(ious)
        all_sizes.append(sizes)
        all_match_sizes.append(match_sizes)

        ious = torch.Tensor(ious)
        y0_ccids = torch.Tensor(y0_ccids)
        y1_ccid_matches = torch.Tensor(y1_ccid_matches)

        # find type 2: areas where connected components have a poor overlap IOU of 0.5 and below but not 0.
        # IOU of zero ccs have already been discounted in the previous step.
        low_ious = ious < 0.5
        low_IOU_y0_ccs = y0_ccids[low_ious]
        low_IOU_y1_ccs = y1_ccid_matches[low_ious]
        low_IOU_y1_ccs = torch.unique(low_IOU_y1_ccs)
        combined_low_overlap_image = create_overlap_image(low_IOU_y0_ccs, low_IOU_y1_ccs, ccs_y0, ccs_y1).type(torch.int)
        combined_low_overlap_diff_image = combined_low_overlap_image * diff_image

        # find type 3: high overlap
        high_ious = (ious >= 0.5) * (ious < 1)
        high_IOU_y0_ccs = y0_ccids[high_ious]
        high_IOU_y1_ccs = y1_ccid_matches[high_ious]
        high_IOU_y1_ccs = torch.unique(high_IOU_y1_ccs)
        combined_high_overlap_image = create_overlap_image(high_IOU_y0_ccs, high_IOU_y1_ccs, ccs_y0, ccs_y1).type(torch.int)
        combined_high_overlap_diff_image = combined_high_overlap_image * diff_image

        # find type 4: areas of exact overlap
        exact_ious = ious == 1
        exact_IOU_y0_ccs = y0_ccids[exact_ious]
        exact_IOU_y1_ccs = y1_ccid_matches[exact_ious]
        exact_IOU_y1_ccs = torch.unique(exact_IOU_y1_ccs)
        combined_exact_overlap_image = create_overlap_image(exact_IOU_y0_ccs, None, ccs_y0, ccs_y1).type(torch.int)

        # then get the areas where the raters agree and remove those areas. see what we are left with...
        # that is the joint uncertainty error overlap... but I want to look at the overlap away from the boundaries....

        # my concern is that we still have all of this edge information.

        # ANALYSIS 1 cc average uncertainty information
        # we need average uncertainty per connected component in the following regions
        # - IR IOU 0
        # - IR IOU 1
        # - diff of IR 0 < IOU < 0.5 regions
        # - diff of IR 0.5 <= IOU < 1 regions

        # ANALYSIS 1.5 pixel wise information
        # the same as analysis 1 but with individual voxels
        ef = e.flatten()
        ent_no_overlap, ent_low_overlap, ent_high_overlap, ent_exact_overlap = ef[combined_no_overlap_image.flatten()==1], ef[combined_low_overlap_diff_image.flatten()==1], ef[combined_high_overlap_diff_image.flatten()==1], ef[combined_exact_overlap_image.flatten()==1]
        ent_no_overlap_all.extend(ent_no_overlap.cpu())
        ent_low_overlap_all.extend(ent_low_overlap.cpu())
        ent_high_overlap_all.extend(ent_high_overlap.cpu())
        ent_exact_overlap_all.extend(ent_exact_overlap.cpu())


        # Analysis 2 connected component extra analysis
        # as the uncertainty threshold increases we look at:
        # average proportion of IR IOU 0 that are uncertain
        # average proportion of IR low IOU that are uncertain
        # average proportion of IR IOU high that are uncertain
        # average proportion of IR IOU 1 that are uncertain
        # number of connected components in uncertainty map that have zero overlap with either rater. I should do this for a fewer number of connected components
        none_prop_uncert = [[] for _ in range(len(uncertainty_thresholds))]
        low_prop_uncert = [[] for _ in range(len(uncertainty_thresholds))]
        high_prop_uncert = [[] for _ in range(len(uncertainty_thresholds))]
        exact_prop_uncert = [[] for _ in range(len(uncertainty_thresholds))]
        non_useful_ccs = [[] for _ in range(len(uncertainty_thresholds))]
        num_FP_uncertainty_ccs = []
        mean_size_FP_uncertainty_ccs = []

        ccs_none = torch.from_numpy(cc3d.connected_components(combined_no_overlap_image.type(torch.int32).cpu().numpy(), connectivity=26).astype(np.float32)).cuda()
        ccs_low = torch.from_numpy(cc3d.connected_components(combined_low_overlap_diff_image.type(torch.int32).cpu().numpy(), connectivity=26).astype(np.float32)).cuda()
        ccs_high = torch.from_numpy(cc3d.connected_components(combined_high_overlap_diff_image.type(torch.int32).cpu().numpy(), connectivity=26).astype(np.float32)).cuda()
        ccs_exact = torch.from_numpy(cc3d.connected_components(combined_exact_overlap_image.type(torch.int32).cpu().numpy(), connectivity=26).astype(np.float32)).cuda()

        any_prediction_image_dist_map = get_distance_map(any_prediction_image)
        any_prediction_image_dist_map = any_prediction_image_dist_map > 5

        # print(ccs_none.unique())
        # print(combined_no_overlap_image.sum())
        for ti, t in enumerate(uncertainty_thresholds):
            et = (e > t).type(torch.float32)

            # look at how many ccs there are
            non_overlapping_uncert_region = et * any_prediction_image_dist_map
            non_overlapping_uncert_ccs = torch.from_numpy(cc3d.connected_components(non_overlapping_uncert_region.type(torch.int32).cpu().numpy(), connectivity=26).astype(np.float32)).cuda()
            num_FP_uncertainty_ccs.append(len(non_overlapping_uncert_ccs.unique()) - 1) # -1 to get rid of background class
            mean_size_FP_uncertainty_ccs.append(
                torch.Tensor(
                    [(non_overlapping_uncert_ccs == cc).sum().item() for cc in non_overlapping_uncert_ccs.unique() if cc != 0]
                ).mean()
            )

            # find proportion uncertain of each cc in each region
            for cc_id in ccs_none.unique():
                if cc_id == 0:
                    continue
                cc = ccs_none == cc_id
                # print(cc_id)
                # print(cc.sum())
                none_prop_uncert_all[ti].append(et[cc==1].mean().item())
                if ti == 0:
                    none_cc_mean.append(e[cc==1].mean())
            for cc_id in ccs_low.unique():
                if cc_id == 0:
                    continue
                cc = ccs_low == cc_id
                low_prop_uncert_all[ti].append(et[cc==1].mean().item())
                if ti == 0:
                    low_cc_mean.append(e[cc==1].mean())
            for cc_id in ccs_high.unique():
                if cc_id == 0:
                    continue
                cc = ccs_high == cc_id
                high_prop_uncert_all[ti].append(et[cc==1].mean().item())
                if ti == 0:
                    high_cc_mean.append(e[cc==1].mean())
            for cc_id in ccs_exact.unique():
                if cc_id == 0:
                    continue
                cc = ccs_exact == cc_id
                exact_prop_uncert_all[ti].append(et[cc==1].mean().item())
                if ti == 0:
                    exact_cc_mean.append(e[cc==1].mean())

        num_FP_uncertainty_ccs_all.append(num_FP_uncertainty_ccs)
        mean_size_FP_uncertainty_ccs_all.append(mean_size_FP_uncertainty_ccs)

        ### UIRO per region analysis.
        no_overlap_distance_map = get_distance_map(combined_no_overlap_image)
        low_overlap_distance_map = get_distance_map(combined_low_overlap_diff_image)
        high_overlap_distance_map = get_distance_map(combined_high_overlap_diff_image)
        uiro_no_overlap, uiro_low_overlap, uiro_high_overlap = IOU_region_UIRO(e, diff_image, no_overlap_distance_map, low_overlap_distance_map, high_overlap_distance_map)
        uiro_no_overlap_all.append(uiro_no_overlap)
        uiro_low_overlap_all.append(uiro_low_overlap)
        uiro_high_overlap_all.append(uiro_high_overlap)
        
    # return (
#         ent_no_overlap_all, # flat list of pixels
#         ent_low_overlap_all, # flat list of pixels
#         ent_high_overlap_all, # flat list of pixels
#         ent_exact_overlap_all, # flat list of pixels
        
#         none_prop_uncert_all, # uncert_thresh * cc across whole dataset
#         low_prop_uncert_all, # uncert_thresh * cc across whole dataset
#         high_prop_uncert_all, # uncert_thresh * cc across whole dataset
#         exact_prop_uncert_all, # uncert_thresh * cc across whole dataset
        
#         num_FP_uncertainty_ccs_all, # n * uncert_thresh
#         mean_size_FP_uncertainty_ccs_all, # n * uncert_thresh
        
#         none_cc_mean, # cc across whole dataset
#         low_cc_mean, # cc across whole dataset
#         high_cc_mean, # cc across whole dataset
#         exact_cc_mean, # cc across whole dataset
        
#         uiro_no_overlap_all, # n * uncert_thresh
#         uiro_low_overlap_all, # n * uncert_thresh
#         uiro_high_overlap_all # n * uncert_thresh
#     )

    num_FP_uncertainty_ccs_all = torch.Tensor(num_FP_uncertainty_ccs_all)
    mean_size_FP_uncertainty_ccs_all = torch.Tensor(mean_size_FP_uncertainty_ccs_all)
    uiro_no_overlap_all = torch.Tensor(uiro_no_overlap_all)
    uiro_low_overlap_all = torch.Tensor(uiro_low_overlap_all)
    uiro_high_overlap_all = torch.Tensor(uiro_high_overlap_all)
    
    overall_results = {
        **{f"{region}_num_FP_uncertainty_ccs_all_t{t:.2f}":num_FP_uncertainty_ccs_all[:,ti] for ti, t in enumerate(uncertainty_thresholds)},
        **{f"{region}_mean_size_FP_uncertainty_ccs_all_t{t:.2f}":mean_size_FP_uncertainty_ccs_all[:,ti] for ti, t in enumerate(uncertainty_thresholds)},
        **{f"{region}_uiro_no_overlap_all_t{t:.2f}":uiro_no_overlap_all[:,ti] for ti, t in enumerate(uncertainty_thresholds)},
        **{f"{region}_uiro_low_overlap_all_t{t:.2f}":uiro_low_overlap_all[:,ti] for ti, t in enumerate(uncertainty_thresholds)},
        **{f"{region}_uiro_high_overlap_all_t{t:.2f}":uiro_high_overlap_all[:,ti] for ti, t in enumerate(uncertainty_thresholds)},
    }
    
    ent_no_overlap_all = torch.Tensor(ent_no_overlap_all).cpu().numpy()
    ent_low_overlap_all = torch.Tensor(ent_low_overlap_all).cpu().numpy()
    ent_high_overlap_all = torch.Tensor(ent_high_overlap_all).cpu().numpy()
    ent_exact_overlap_all = torch.Tensor(ent_exact_overlap_all).cpu().numpy()
    
    none_cc_mean = torch.Tensor(none_cc_mean).cpu().numpy()
    low_cc_mean = torch.Tensor(low_cc_mean).cpu().numpy()
    high_cc_mean = torch.Tensor(high_cc_mean).cpu().numpy()
    exact_cc_mean = torch.Tensor(exact_cc_mean).cpu().numpy()
    
    pixelwise_and_cc_results = {**{
        f"{region}_ent_no_overlap_all_pixels":ent_no_overlap_all ,
        f"{region}_ent_low_overlap_all_pixels":ent_low_overlap_all ,
        f"{region}_ent_high_overlap_all_pixels":ent_high_overlap_all ,
        f"{region}_ent_exact_overlap_all_pixels":ent_exact_overlap_all ,
        f"{region}_none_cc_mean":none_cc_mean ,
        f"{region}_low_cc_mean":low_cc_mean ,
        f"{region}_high_cc_mean":high_cc_mean ,
        f"{region}_exact_cc_mean":exact_cc_mean ,
    }, 
        **{f"{region}_none_prop_uncert_all_t{t:.2f}":torch.Tensor(none_prop_uncert_all[ti]).cpu().numpy() for ti, t in enumerate(uncertainty_thresholds)},
        **{f"{region}_low_prop_uncert_all_t{t:.2f}":torch.Tensor(low_prop_uncert_all[ti]).cpu().numpy() for ti, t in enumerate(uncertainty_thresholds)},
        **{f"{region}_high_prop_uncert_all_t{t:.2f}":torch.Tensor(high_prop_uncert_all[ti]).cpu().numpy() for ti, t in enumerate(uncertainty_thresholds)},
        **{f"{region}_exact_prop_uncert_all_t{t:.2f}":torch.Tensor(exact_prop_uncert_all[ti]).cpu().numpy() for ti, t in enumerate(uncertainty_thresholds)},
                               }
    
    return overall_results, pixelwise_and_cc_results

def dJUEO(pred, thresholded_umap, IR, IR_distmap, ME, ME_distmap, d_threshold):
    
    dthresholded_umap = thresholded_umap * torch.logical_or(IR_distmap >= d_threshold, ME_distmap < d_threshold)
    
    return fast_dice(dthresholded_umap, ME)

def dUIRO(pred, thresholded_umap, IR, IR_distmap, ME, ME_distmap, d_threshold):
    
    dthresholded_umap = thresholded_umap * torch.logical_or(IR_distmap < d_threshold, ME_distmap >= d_threshold)
    
    return fast_dice(dthresholded_umap, IR)

def soft_dJUEO(pred, umap, IR, IR_distmap, ME, ME_distmap, d_threshold):
    
    dumap = umap * torch.logical_or(IR_distmap >= d_threshold, ME_distmap < d_threshold)
    
    return soft_dice(dumap, ME)

def soft_dUIRO(pred, umap, IR, IR_distmap, ME, ME_distmap, d_threshold):
    
    dumap = umap * torch.logical_or(IR_distmap < d_threshold, ME_distmap >= d_threshold)
    
    return soft_dice(dumap, IR)

def perform_d_UEO_analysis(pred, umap, seg1, seg2, uncertainty_thresholds):
    sdJUEOs = []
    sdUIROs = []
    dUIROs = []
    dJUEOs = []
    
    ir = (seg1 != seg2)
    ir_distmap = get_distance_map(ir)
    
    me = (seg1 == seg2) * (seg1 != pred)
    me_distmap = get_distance_map(me)
    
    # when d_threshold = 0.5, this is equivalent to just masking out the exact pixels.
    distance_thresholds = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]
    for d_threshold in distance_thresholds:
        sdJUEOs.append(soft_dJUEO(pred, umap, ir, ir_distmap, me, me_distmap, d_threshold))
        sdUIROs.append(soft_dUIRO(pred, umap, ir, ir_distmap, me, me_distmap, d_threshold))
        UIRO = []
        JUEO = []
        for t in uncertainty_thresholds:
            tumap = umap > t
            JUEO.append(dJUEO(pred, tumap, ir, ir_distmap, me, me_distmap, d_threshold))
            UIRO.append(dUIRO(pred, tumap, ir, ir_distmap, me, me_distmap, d_threshold))
    
        dJUEOs.append(JUEO)
        dUIROs.append(UIRO)
        
    return sdJUEOs, sdUIROs, dJUEOs, dUIROs, distance_thresholds