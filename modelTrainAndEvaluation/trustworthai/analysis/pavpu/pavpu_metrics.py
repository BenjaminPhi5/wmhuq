import torch
from tqdm import tqdm
from trustworthai.utils.logits_to_preds import normalize_samples

def compute_pavpu_metrics(means3d, samples3d, ys3d, ind_ent_maps, uncertainty_thresholds=torch.arange(0, 0.7, 0.01), accuracy_threshold=0.9, window_size=16, do_normalize=True):
    
    #uncertainty_thresholds = torch.arange(0, 0.7, 0.01)
    #accuracy_threshold = 0.9
    window = window_size#16
    stride = window_size#16
    n_acs = [[] for i in range(len(uncertainty_thresholds))]
    n_aus = [[] for i in range(len(uncertainty_thresholds))]
    n_ics = [[] for i in range(len(uncertainty_thresholds))]
    n_ius = [[] for i in range(len(uncertainty_thresholds))]

    for batch_idx in tqdm(range(len(ys3d)), ncols=150, position=0, leave=True): # skip the last batch with a different shape
        batch_t = ys3d[batch_idx].squeeze()
        batch_samples = samples3d[batch_idx]
        batch_mean = means3d[batch_idx]
        ent = ind_ent_maps[batch_idx]

        # get probabilities
        if do_normalize:
            probs = normalize_samples(batch_samples)
        else:
            probs = batch_samples
        pred_classes = probs.argmax(dim=2)
        confidences = probs.max(dim=2)[0]

        # get average accuracy of each sample using the mean
        # or I could treat each patch of each sample as a separate thing but that is not what I am doing here.
        #avg_accuracy = ((batch_t.expand(pred_classes.shape) == pred_classes) * 1.).mean(dim=0)
        accuracy = batch_t == batch_mean.argmax(dim=1)

        # unroll predictions and targets and entropy
        t_unrolled = batch_t.unfold(-2, window, stride).unfold(-1, window, stride).reshape(-1, window, window)
        #accuracy_unrolled = avg_accuracy.unfold(-2, window, stride).unfold(-1, window, stride).reshape(-1, window, window)
        accuracy_unrolled = accuracy.unfold(-2, window, stride).unfold(-1, window, stride).reshape(-1, window, window)
        ent_unrolled = ent.unfold(-2, window, stride).unfold(-1, window, stride).reshape(-1, window, window)

        #accurate_patches = accuracy_unrolled > 0.9
        accurate_patches = accuracy_unrolled.type(torch.float32).mean(dim=(1,2)) > 0.9
        # print(accurate_patches.shape)

        # try applying around patches that have lesion burden.
        has_lesion = t_unrolled.mean(dim=(1,2)) > 0.

        # for each uncertainty threshold, compute the 4 numbers
        for i, uncert_t in enumerate(uncertainty_thresholds):
            #uncertain_patches = ent_unrolled > uncert_t
            uncertain_patches = ent_unrolled.mean(dim=(1,2)) > uncert_t
            # print(uncertain_patches.shape)

            n_acs[i].append(torch.sum(torch.logical_and(has_lesion, torch.logical_and(accurate_patches, ~uncertain_patches))))
            n_aus[i].append(torch.sum(torch.logical_and(has_lesion, torch.logical_and(accurate_patches, uncertain_patches))))
            n_ics[i].append(torch.sum(torch.logical_and(has_lesion, torch.logical_and(~accurate_patches, ~uncertain_patches))))
            n_ius[i].append(torch.sum(torch.logical_and(has_lesion, torch.logical_and(~accurate_patches, uncertain_patches))))

    n_acs_t = [torch.Tensor(n_acs[i]).sum() for i in range(len(uncertainty_thresholds))]
    n_aus_t = [torch.Tensor(n_aus[i]).sum() for i in range(len(uncertainty_thresholds))]
    n_ics_t = [torch.Tensor(n_ics[i]).sum() for i in range(len(uncertainty_thresholds))]
    n_ius_t = [torch.Tensor(n_ius[i]).sum() for i in range(len(uncertainty_thresholds))]

    p_acs = [n_acs_t[i] / (n_acs_t[i] + n_ics_t[i]) for i in range(len(uncertainty_thresholds))]
    p_aus = [n_ius_t[i] / (n_ius_t[i] + n_ics_t[i]) for i in range(len(uncertainty_thresholds))]
    pavpu = [(n_acs_t[i] + n_ius_t[i]) / (n_ius_t[i] + n_ics_t[i] + n_aus_t[i] + n_acs_t[i]) for i in range(len(uncertainty_thresholds))]
    
    return p_acs, p_aus, pavpu
    
    
