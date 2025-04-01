import matplotlib.pyplot as plt
from trustworthai.utils.plotting.saving_plots import save
from trustworthai.utils.print_and_write_func import print_and_write
import torch
from trustworthai.analysis.pavpu.pavpu_metrics import compute_pavpu_metrics

# I am going to do it per patch, but take the average accuracy per patch (perhaps I should qc average dice as well, best dice, worst dice.
def pavpu_analysis(save_folder, text_results_file, means3d, samples3d, ys3d, ind_ent_maps, uncertainty_thresholds=torch.arange(0, 0.7, 0.01), accuracy_threshold=0.9, window_size=16, do_normalize=True):
    p_acs, p_aus, pavpu = compute_pavpu_metrics(means3d, samples3d, ys3d, ind_ent_maps, uncertainty_thresholds, accuracy_threshold, window_size, do_normalize)
    record_pavpu_results(save_folder, text_results_file, uncertainty_thresholds, p_acs, p_aus, pavpu)

def record_pavpu_results(save_folder, text_results_file, uncertainty_thresholds, p_acs, p_aus, pavpu):
    plt.figure(figsize=(13,3))
    plt.subplot(1,3,1)
    plt.plot(uncertainty_thresholds, p_acs, c='g')
    plt.xlim((-0.01,0.7)); plt.ylim((-0.05,1.05))
    plt.xlabel("τ")
    plt.ylabel("p(acc|cert)")
    plt.subplot(1,3,2)
    plt.plot(uncertainty_thresholds, p_aus, c='g')
    plt.xlim((-0.01,0.7)); plt.ylim((-0.05,1.05))
    plt.ylabel("p(uncert|inacc)")
    plt.xlabel("τ")
    plt.subplot(1,3,3)
    plt.plot(uncertainty_thresholds, pavpu, c='g')
    plt.xlim((-0.01,0.7)); plt.ylim((-0.05,1.05))
    plt.ylabel("PAVPU")
    plt.xlabel("τ")
    save(save_folder, "pavpu")

    print_and_write(text_results_file, f"p_acs:", newline=1)
    print_and_write(text_results_file, torch.stack(p_acs))
    print_and_write(text_results_file, f"p_aus:", newline=1)
    print_and_write(text_results_file, torch.stack(p_aus))
    print_and_write(text_results_file, f"pavpu:", newline=1)
    print_and_write(text_results_file, torch.stack(pavpu))
    
    