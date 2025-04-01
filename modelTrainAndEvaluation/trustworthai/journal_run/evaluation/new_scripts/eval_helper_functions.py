import torch
import pandas as pd
from trustworthai.analysis.evaluation_metrics.challenge_metrics import do_challenge_metrics
from tqdm import tqdm
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import cc3d

from trustworthai.utils.fitting_and_inference.fitters.basic_lightning_fitter import StandardLitModelWrapper
from trustworthai.utils.fitting_and_inference.fitters.p_unet_fitter import PUNetLitModelWrapper

def load_best_checkpoint(model, loss, model_ckpt_folder, punet=False):
    # this is ultimately going to need to be passed a model wrapper when I implement P-Unet....
    
    # the path to the best checkpoint is stored as a single line in a txt file along with each model
    with open(os.path.join(model_ckpt_folder, "best_ckpt.txt"), "r") as f:
        ckpt_file = os.path.join(model_ckpt_folder, f.readlines()[0][:-1].split("/")[-1])
    
    # try:
    if punet:
        return PUNetLitModelWrapper.load_from_checkpoint(ckpt_file, model=model, loss=loss, 
                                    logging_metric=lambda : None)
    return StandardLitModelWrapper.load_from_checkpoint(ckpt_file, model=model, loss=loss, 
                                    logging_metric=lambda : None)
    # except:
    #     raise ValueError(f"ckpt {ckpt_file} couldn't be loaded, maybe it doesn't exist?")


def per_model_chal_stats(preds3d, ys3d):
    stats = []
    for i in tqdm(range(len(ys3d)), position=0, leave=True):
        ind_stats = do_challenge_metrics(ys3d[i].type(torch.long), preds3d[i].argmax(dim=1).type(torch.long))
        stats.append(ind_stats)

    tstats = torch.Tensor(stats)
    dices = tstats[:,0]
    hd95s = tstats[:,1]
    avds = tstats[:,2]
    recalls = tstats[:,3]
    f1s = tstats[:,4]

    data = {"dice":dices, "hd95":hd95s, "avd":avds, "recall":recalls, "f1":f1s}

    return data

def write_per_model_channel_stats(preds, ys3d_test, args, chal_results=None, fname='overall_stats.csv', individual_stats_name='individual_stats.csv'):
    if chal_results == None:
        chal_results = per_model_chal_stats(preds, ys3d_test)
    
    model_result_dir = os.path.join(args.repo_dir, args.result_dir, f"UNCERT_TYPE_{args.uncertainty_type}_base_model_type_{args.model_name}")
    
    # save the results to pandas dataframes
    # for key, value in chal_results.items():
    #     print(key, ": ", len(value))
    df = pd.DataFrame(chal_results)
    df['model_name'] = [args.model_name for _ in range(len(df))]
    
    df.to_csv(model_result_dir + f"{individual_stats_name}.csv")
    
    overall_stats = {"model_name":[args.model_name]}

    for key, value in chal_results.items():
        print(key)
        mean = value.mean()
        std = value.std(correction=1) # https://en.wikipedia.org/wiki/Bessel%27s_correction#Source_of_bias in this case we know the true mean..?
        # I want the standard deviation across this dataset, and I have a full sample, so I should use correction = 0? Or are we saying we have a limited sample of data from
        # an infinite distribution, and we want to know the model performance on that distribution, so correction = 1. Hmm this is a bit of a headache.
        conf_int = 1.96 * std / np.sqrt(len(value))

        lower_bound = mean - conf_int
        upper_bound = mean + conf_int

        overall_stats[f"{key}_mean"] = [mean.item()]
        overall_stats[f"{key}_95%conf"] = [conf_int.item()]
        
    overall_stats = pd.DataFrame(overall_stats)
    
    overall_stats.to_csv(model_result_dir + f"_{fname}")
    
    
def per_sample_metric(samples, ys3d_test, f, do_argmax, do_softmax, minimise, take_abs=False):
    num_samples = len(samples[0])
    samples_f = []
    for i in tqdm(range(len(ys3d_test)), position=0, leave=True):
        scores = []
        y = ys3d_test[i].cuda()
        for j in range(num_samples):
            if do_argmax:
                s = samples[i][j].cuda().argmax(dim=1)
            elif do_softmax:
                s = torch.softmax(samples[i][j].cuda(), dim=1)
            else:
                s = samples[i][j].cuda()
            scores.append(f(s, y))
        scores = torch.Tensor(scores)
        samples_f.append(scores)

    samples_f = torch.stack(samples_f)

    if take_abs:
        samples_optimize = samples_f.abs()
    else:
        samples_optimize = samples_f
    
    if minimise:
        return samples_optimize.min(dim=1)[0], samples_f
    else:
        return samples_optimize.max(dim=1)[0], samples_f
    

def fast_rmse(pred, y, p=0.1):
    locs = (pred[:,1] > p) | (y == 1)
    onehot = (y.unsqueeze(dim=1)==y.unique().view(1, -1, 1, 1)).type(torch.float32)
    
    pred = pred.moveaxis(1, -1)[locs]
    onehot = onehot.moveaxis(1, -1)[locs]
    
    # should this not be a sum on dim -1 given the line above!!!!!!
    return (pred - onehot).square().sum(dim=1).mean().sqrt()


def fast_dice(pred, target):
    p1 = (pred == 1)
    t1 = (target == 1)
    intersection = (pred == 1) & (target == 1)
    numerator = 2 * intersection.sum()
    denominator = p1.sum() + t1.sum()
    return (numerator/(denominator + 1e-30)).item()

def fast_recall(pred, target):
    p1 = (pred == 1)
    t1 = (target == 1)
    return ((p1 * t1).sum() / (t1.sum() + 1e-30)).item()

def fast_precision(pred, target):
    p1 = (pred == 1)
    t1 = (target == 1).type(torch.int32)
    tp = (p1 * t1).sum()
    return (tp / (tp + (p1 * (1-t1)).sum() + 1e-30)).item()

def fast_fpr(pred, target):
    p1 = (pred == 1)
    t1 = (target == 1).type(torch.int32)
    
    return ((p1 * (1-t1)).sum()/((1-t1).sum() + 1e-30)).item()


def fast_avd(pred, target):
    p1 = pred.sum()
    t1 = target.sum()
    
    return (((p1 - t1).abs() / t1) * 100).item()

def fast_vd(pred, target):
    p1 = pred.sum()
    t1 = target.sum()
    
    return (((p1 - t1) / t1) * 100).item()


def get_xs_and_ys(eval_ds):
    xs3d_test = []
    ys3d_test = []

    for i, data in enumerate(eval_ds):
        ys3d_test.append(data[1].squeeze())
        xs3d_test.append(data[0])
        
    return xs3d_test, ys3d_test

def GT_volumes(ys3d):
    volumes = []
    for y in ys3d:
        volumes.append(y.sum())
    return torch.Tensor(volumes)

def fast_iou(pred, target):
    p1 = (pred == 1)
    t1 = (target == 1)
    intersection = (pred == 1) & (target == 1)
    numerator = intersection.sum()
    denominator = p1.sum() + t1.sum() - numerator
    return (numerator/(denominator + 1e-30)).item()

def iou_GED(means, ys3d_test, samples):
    geds = []
    
    for i in tqdm(range(len(means)), position=0, leave=True):
        y = ys3d_test[i].cuda()
        ss = samples[i].cuda().argmax(dim=2)
        num_samples = ss.shape[0]
        
        dists_ab = 0
        
        # print(y.sum())
        
        for s in ss:
            pred = s#.argmax(dim=1)
            dists_ab += (1 - fast_iou(pred, y))
            # print(dists_ab)
            # print(s.shape)
        
        dists_ab /= num_samples
        dists_ab *= 2
        
        dists_ss = 0
        for j, s1 in enumerate(ss):
            for k, s2 in enumerate(ss):
                if j == k:
                    continue
                dists_ss += (1 - fast_iou(s1, s2))
        
        dists_ss /= (num_samples * (num_samples - 1))
        
        ged = dists_ab - dists_ss
        if not np.isnan(ged):
            geds.append(ged)
        #break
        
    return torch.Tensor(geds)

def individual_pavpu(mean, ent_map, label, ws=4, acc_t=0.8, uncertainty_thresholds=None):
    pred_unfolded = mean.argmax(dim=1).unfold(0, ws, ws).unfold(1, ws, ws).unfold(2, ws, ws).reshape(-1, ws, ws, ws)
    ent_unfolded = ent_map.unfold(0, ws, ws).unfold(1, ws, ws).unfold(2, ws, ws).reshape(-1, ws, ws, ws)
    ys_unfolded = label.unfold(0, ws, ws).unfold(1, ws, ws).unfold(2, ws, ws).reshape(-1, ws, ws, ws)

    patch_acc = (pred_unfolded == ys_unfolded).type(torch.float32).mean(dim=(1,2,3))
    patch_uncert = ent_unfolded.mean(dim=(1,2,3))
    patch_non_empty = ys_unfolded.sum(dim=(1,2,3)) > 0

    p_acc_cert = []
    p_uncert_inacc = []
    pavpu = []

    for tau in uncertainty_thresholds:
        acc = (patch_acc >= acc_t) * patch_non_empty
        inacc = (patch_acc < acc_t) * patch_non_empty
        cert = (patch_uncert < tau) * patch_non_empty
        uncert = (patch_uncert >= tau) * patch_non_empty

        n_ac = (acc * cert).sum().item()
        n_au = (acc * uncert).sum().item()
        n_iu = (inacc * uncert).sum().item()
        n_ic = (inacc * cert).sum().item()

        p_acc_cert.append(n_ac / (n_ac + n_ic + 1e-30))
        p_uncert_inacc.append(n_iu / (n_ic + n_iu + 1e-30))
        pavpu.append((n_ac + n_iu) / (n_ac + n_au + n_ic + n_iu + 1e-30))
        
    return p_acc_cert, p_uncert_inacc, pavpu

def all_individuals_pavpu(means, ent_maps, ys, window_size=4, accuracy_threshold=0.8, uncertainty_thresholds=None):
    all_acc_cert = []
    all_uncert_inacc = []
    all_pavpu = []
    for i in tqdm(range(len(means)), position=0, leave=True):
        p_acc_cert, p_uncert_inacc, pavpu = individual_pavpu(means[i].cuda(), ent_maps[i].cuda(), ys[i].cuda(), window_size, accuracy_threshold, uncertainty_thresholds)
        all_acc_cert.append(p_acc_cert)
        all_uncert_inacc.append(p_uncert_inacc)
        all_pavpu.append(pavpu)
        # break
    
    return torch.Tensor(all_acc_cert), torch.Tensor(all_uncert_inacc), torch.Tensor(all_pavpu)

def sUEO(pred, ent_map, target):
    errors = (pred != target)
    
    numerator = 2 * (ent_map * errors).sum()
    denominator = (errors**2).sum() + (ent_map**2).sum()
    
    return (numerator / denominator).item()

def get_sUEOs(means, ys3d_test, ent_maps):
    sUEOs = []
    for i in tqdm(range(len(ent_maps)), position=0, leave=True):
        pred = means[i].argmax(dim=1).cuda()
        target = ys3d_test[i].cuda()
        ent = ent_maps[i].cuda()

        # if pred.sum() == 0:
        #     continue

        sUEOs.append(sUEO(pred, ent, target))

    sUEOs = torch.Tensor(sUEOs)
    return sUEOs

def UEO_per_threshold_analysis(uncertainty_thresholds, ys3d, ind_ent_maps, means, max_ent):
    ueos = [[] for _ in range(len(uncertainty_thresholds))]
                              
    for i in tqdm(range(len(ys3d)), position=0, leave=True):
        pred = means[i].argmax(dim=1).cuda()
        target = ys3d[i].cuda()
        ent = ind_ent_maps[i].cuda()
        
        # if pred.sum() == 0:
        #     continue
        
        for j, t in enumerate((uncertainty_thresholds)):
            ueos[j].append(sUEO(pred, (ent > t).type(torch.float32), target))
    
    ueos = torch.stack([torch.Tensor(ind_ueo) for ind_ueo in ueos], dim=0)

    return ueos


def UEO_per_threshold_recall_precision(uncertainty_thresholds, ys3d, ind_ent_maps, means, max_ent):
    ueos_recall = [[] for _ in range(len(uncertainty_thresholds))]
    ueos_precision = [[] for _ in range(len(uncertainty_thresholds))]
                              
    for i in tqdm(range(len(ys3d)), position=0, leave=True):
        pred = means[i].argmax(dim=1).cuda()
        target = ys3d[i].cuda()
        ent = ind_ent_maps[i].cuda()
        
        errors = (pred != target)
        
        for j, t in enumerate((uncertainty_thresholds)):
            ueos_recall[j].append(fast_recall(ent > t, errors))
            ueos_precision[j].append(fast_precision(ent > t, errors))
    
    ueos_recall = torch.stack([torch.Tensor(ind_ueo) for ind_ueo in ueos_recall], dim=0)
    ueos_precision = torch.stack([torch.Tensor(ind_ueo) for ind_ueo in ueos_precision], dim=0)

    return ueos_recall, ueos_precision


def get_3d_cc_analysis(img, mean, ent, uncertainty_thresholds, ss=5, prop_size=0.5):
    mean = mean.argmax(dim=1).cuda()
    ent = ent.cuda()
    labels_in = img.cpu()
    labels_out = cc3d.connected_components(img.type(torch.int32).numpy(), connectivity=26) # 26-connected
    labels_out = torch.from_numpy(labels_out.astype(np.float32)).cuda()
    
    c_thresholds = [t.item() for t in uncertainty_thresholds]

    num_lesions = labels_out.unique().shape[0] - 1
    sizes = []
    missing_area_sizes = []
    missing_area_coverage = {ct:[] for ct in c_thresholds}
    proportion_missing_lesion_covered_ent = {ct:[] for ct in c_thresholds}
    num_entirely_missed_lesions = {ct:0 for ct in c_thresholds}
    entirely_missed_lesions_size = {ct:[] for ct in c_thresholds}

    for ccid in labels_out.unique():
        if ccid == 0:
            continue

        cc = labels_out == ccid
        size = cc.sum().item()
        sizes.append(size)

        missing_area = (mean == 0) & cc
        ma_size = missing_area.sum()
        # print(ma_size)
        missing_area_sizes.append(ma_size)

        # get uncertain pixels for each threshold
        for tau in c_thresholds:
            uncert = (ent > tau).type(torch.long)

            # coverage proportion
            coverage = (uncert * missing_area).sum() / (ma_size + 1e-30)
            missing_area_coverage[tau].append(coverage)


            ss = 5
            prop_ss = 0.5
            if torch.sum(mean * cc) <= min(size * prop_ss, ss): # (size * prop_ss):
                # proportion of those lesions that are missing from mean covered by uncertainty
                proportion_missing_lesion_covered_ent[tau].append(torch.sum(uncert * cc) / size)

                # lesions entirely missed by both mean prediction and uncertainty map
                # i.e not a single voxel is identified as uncertain or mean, total silent failure.
                if torch.sum(uncert * cc) <= min(size * prop_ss, ss): # (size * prop_ss):
                    num_entirely_missed_lesions[tau] += 1
                    entirely_missed_lesions_size[tau].append(size)

    mean_missed_area3d = torch.Tensor([torch.Tensor(missing_area_coverage[tau]).mean().item() for tau in c_thresholds])
    mean_size_missed_lesions3d = torch.Tensor([torch.Tensor(entirely_missed_lesions_size[tau]).mean().item() for tau in c_thresholds])
    mean_cov_mean_missed_lesions3d = torch.Tensor([torch.Tensor(proportion_missing_lesion_covered_ent[tau]).mean().item() for tau in c_thresholds])
    # num_missed_lesions3d = torch.Tensor([num_entirely_missed_lesions[tau] for tau in c_thresholds])
    prop_lesions_missed3d = torch.Tensor([num_entirely_missed_lesions[tau]/num_lesions for tau in c_thresholds])

    return num_lesions, sizes, mean_missed_area3d, mean_size_missed_lesions3d, mean_cov_mean_missed_lesions3d, prop_lesions_missed3d


def do_3d_cc_analysis_per_individual(means, ys3d_test, ent_maps, uncertainty_thresholds):
    num_lesions_all, sizes_all, mean_missed_area3d_all, mean_size_missed_lesions3d_all, mean_cov_mean_missed_lesions3d_all, prop_lesions_missed3d_all = [], [], [], [], [], []
    for i in tqdm(range(len(means)), position=0, leave=True):
        num_lesions, sizes, mean_missed_area3d, mean_size_missed_lesions3d, mean_cov_mean_missed_lesions3d, prop_lesions_missed3d = get_3d_cc_analysis(ys3d_test[i], means[i], ent_maps[i], uncertainty_thresholds)
        num_lesions_all.append(num_lesions)
        sizes_all.append(sizes)
        mean_missed_area3d_all.append(mean_missed_area3d)
        mean_size_missed_lesions3d_all.append(mean_size_missed_lesions3d)
        mean_cov_mean_missed_lesions3d_all.append(mean_cov_mean_missed_lesions3d)
        prop_lesions_missed3d_all.append(prop_lesions_missed3d)
        
    return num_lesions_all, sizes_all, mean_missed_area3d_all, mean_size_missed_lesions3d_all, mean_cov_mean_missed_lesions3d_all, prop_lesions_missed3d_all

def fast_slice_dice(pred, target):
    bs = pred.shape[0]
    pred = pred.view(bs, -1)
    target = target.view(bs, -1)
    p1 = (pred == 1)
    t1 = (target == 1)
    intersection = (pred == 1) & (target == 1)
    numerator = 2 * intersection.sum(dim=1) + 1e-30
    denominator = p1.sum(dim=1) + t1.sum(dim=1)
    return (numerator/(denominator + 1e-30))

def reorder_samples_by_dice(sample, ys):
    sample = sample.cuda()
    ys = ys.cuda()
    preds = sample.argmax(dim=2)

    sample_dice = torch.stack([fast_slice_dice(s, ys) for s in preds])

    slice_dice_orders = torch.sort(sample_dice.T, dim=1)[1]

    # rearrange the samples into one...
    new_sample = torch.zeros(sample.shape).to(sample.device)
    for i, slice_dice_orders in enumerate(slice_dice_orders):
        for j, sample_index in enumerate(slice_dice_orders):
            new_sample[j][i] = sample[sample_index][i]

    new_sample = new_sample.cpu()
    
    return new_sample