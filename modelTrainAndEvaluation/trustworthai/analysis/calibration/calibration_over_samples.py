import torch
import matplotlib.pyplot as plt
from trustworthai.utils.plotting.saving_plots import save
from trustworthai.utils.print_and_write_func import print_and_write
from trustworthai.analysis.calibration.helper_funcs import *
from tqdm import tqdm
from trustworthai.utils.logits_to_preds import normalize_samples


def calibration_over_samples(save_folder, results_file, means3d, samples3d, ys3d, do_normalize):
    bins = 10 + 1 # for the 0 bin
    bin_batch_accuracies = [[] for b in range(bins)]
    bin_batch_confidences = [[] for b in range(bins)]
    bin_batch_sizes = [[] for b in range(bins)]
    bin_counts = [0 for b in range(bins)]
    for batch_idx in tqdm(range(len(ys3d)), ncols=150, position=0, leave=True): # skip the last batch with a different shape
        batch_t = ys3d[batch_idx].squeeze()
        batch_samples = samples3d[batch_idx]

        if batch_t.shape[0] < 10:
            continue # skip last batch if it is very small.

        # get probabilities
        if do_normalize:
            probs = normalize_samples(batch_samples)
        else:
            probs = batch_samples
        p1s = probs[:,:,1]

        # split into bins
        bin_ids = place_in_bin(p1s)

        # compute counts
        for i in range(bins):
            is_in_bin = (bin_ids == (i / 10))
            # print(is_in_bin.shape)
            # print(batch_t.shape)

            # number of elements in each bin
            num_elem = torch.sum(is_in_bin).item()
            if num_elem == 0:
                print("zero")

            # number of predictions = to class 1
            c1_acc = batch_t.expand(p1s.shape)[is_in_bin].sum() / num_elem

            if torch.isnan(c1_acc):
                print("acc_nan")

            # average confidence of values in that bin
            c1_conf = p1s[is_in_bin].mean()

            if torch.isnan(c1_conf):
                print("conf_nan")
                
            if torch.isnan(c1_conf) or torch.isnan(c1_acc) or num_elem == 0:
                #print("conf_nan") # just skip for this bin for this indivudal if they don't have have a prediction
                # with a confidence in this bin.
                continue

            bin_batch_accuracies[i].append(c1_acc)
            bin_batch_confidences[i].append(c1_conf)
            bin_batch_sizes[i].append(num_elem)

    bin_sizes = [torch.Tensor(bbs).sum() for bbs in bin_batch_sizes]
    bin_accuracies = [torch.Tensor([bin_batch_accuracies[i][j] * bin_batch_sizes[i][j] / bin_sizes[i] for j in range(len(bin_batch_accuracies[i]))]).sum().item() for i in range(len(bin_sizes))]
    bin_confidences = [torch.Tensor([bin_batch_confidences[i][j] * bin_batch_sizes[i][j] / bin_sizes[i] for j in range(len(bin_batch_confidences[i]))]).sum().item() for i in range(len(bin_sizes))]

    print_and_write(results_file, "calibration curve data: ")

    print_and_write(results_file, "bin_accuracies: ", newline=1)
    print_and_write(results_file, str(bin_accuracies))

    print_and_write(results_file, "bin_confidences: ", newline=1)
    print_and_write(results_file, str(bin_confidences))

    total_size = torch.sum(torch.Tensor(bin_sizes)[1:])
    ece = torch.sum( (torch.Tensor(bin_sizes)[1:]/ total_size) * (torch.abs(torch.Tensor(bin_accuracies)[1:] - torch.Tensor(bin_confidences)[1:])))
    print_and_write(results_file, f"EXPECTED CALIBRATION ERROR", newline=1)
    print("note we skip the first bin due to its size")
    print_and_write(results_file, ece)

    plt.plot(bin_confidences, bin_accuracies)
    plt.plot([0,1],[0,1]);
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy");
    save(save_folder, "calibration")