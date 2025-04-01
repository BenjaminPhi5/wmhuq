from trustworthai.utils.print_and_write_func import print_and_write
from trustworthai.utils.plotting.saving_plots import save
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn import metrics


def sUEO(ent, target):
    numerator = 2 * torch.sum(ent * target)
    denominator = torch.sum((target**2) + (ent**2))
    return (numerator / denominator).item()


def sUEO_per_individual_analysis(text_results_file, ys3d, ind_ent_maps):
    sueo_s = []
    for i in range(len(ys3d)):
        sueo_s.append(sUEO(ind_ent_maps[i], ys3d[i]))
        
    print_and_write(text_results_file, f"sUEO", newline=1)
    print_and_write(text_results_file, torch.mean(torch.Tensor(sueo_s)).item())
    
def UEO_per_threshold_analysis(save_folder, text_results_file, uncertainty_thresholds, ys3d, ind_ent_maps):
    ueos = []
    for t in tqdm(uncertainty_thresholds, position=0, leave=True):
        t_ueos = []
        for i in range(len(ys3d)):
            t_ueos.append((sUEO((ind_ent_maps[i] > t).type(torch.float32), ys3d[i])))
        ueos.append(torch.Tensor(t_ueos).mean().item())

    best_index = torch.Tensor(ueos).argmax()
    print_and_write(text_results_file, f"best tau for max UEO", newline=1)
    print_and_write(text_results_file, uncertainty_thresholds[best_index])
    print_and_write(text_results_file, "max UEO", newline=1)
    print_and_write(text_results_file, ueos[best_index])

    print_and_write(text_results_file, f"UEO per tau", newline=1)
    print_and_write(text_results_file, torch.Tensor(ueos))


    plt.plot(uncertainty_thresholds, ueos)
    plt.xlabel("τ")
    plt.ylabel("UEO")
    save(save_folder, "UEO")

    print_and_write(text_results_file, f"tau AUC", newline=1)
    print_and_write(text_results_file, metrics.auc(uncertainty_thresholds, ueos))