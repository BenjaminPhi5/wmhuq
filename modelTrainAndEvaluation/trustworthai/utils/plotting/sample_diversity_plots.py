import torch
import matplotlib.pyplot as plt
from trustworthai.utils.plotting.saving_plots import save


def sample_diversity_plot(save_folder, sample_metrics_3d, metric_name):
    # sort in order of quality
    order = torch.sort(torch.median(sample_metrics_3d, dim=0)[0])[1]
    plt.figure(figsize=(20, 5))
    plt.boxplot(sample_metrics_3d.T[order]);
    plt.ylim(0, 1);
    plt.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False) # labels along the bottom edge are off
    plt.ylabel(f"{metric_name}")
    plt.xlabel("Individuals")
    save(save_folder, "sample_diversity_plot")