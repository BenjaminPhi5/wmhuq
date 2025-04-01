import torch
from trustworthai.analysis.evaluation_metrics.IOU_3D import all_samples_iou
from trustworthai.utils.print_and_write_func import print_and_write
from tqdm import tqdm

def sample_diversity(samples):
    samples = samples.argmax(dim=2)
    ss = samples.shape[0]
    samples = samples.view(ss, -1)
    rolled = samples
    diversities = []
    for i in range(2):
        rolled = torch.roll(rolled, 1, 0)
        diversities.append(all_samples_iou(samples, rolled))
    return 1. - torch.mean(torch.Tensor(diversities))

def sample_to_target_distance(samples, target):
    samples = samples.argmax(dim=2)
    target = target.unsqueeze(0).expand(samples.shape)
    ns = samples.shape[0]
    ind_distance = 1. - all_samples_iou(samples.view(ns, -1),target.view(ns, -1))
    return ind_distance


def sample_diverity_analysis(text_results_file, samples3d, ys3d):
    per_ind_diversities = []
    for s in tqdm(samples3d[::2], position=0, leave=True):
        per_ind_diversities.append(sample_diversity(s))

    overall_sample_diversity = torch.Tensor(per_ind_diversities).mean()
    print_and_write(text_results_file, "sample diversity", newline=1)
    print_and_write(text_results_file, overall_sample_diversity)

    all_imgs_stot_distance = []
    for i in tqdm(range(len(ys3d)), position=0, leave=True):
        all_imgs_stot_distance.append(sample_to_target_distance(samples3d[i], ys3d[i]))

    print_and_write(text_results_file, f"generalized energy distance:", newline=1)
    print_and_write(text_results_file, 2 * torch.Tensor(all_imgs_stot_distance).mean() - overall_sample_diversity)
    #2 * torch.Tensor(all_imgs_stot_distance).mean() - overall_sample_diversity