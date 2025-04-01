import torch
import matplotlib.pyplot as plt
from trustworthai.utils.plotting.saving_plots import save
from trustworthai.utils.print_and_write_func import print_and_write
from tqdm import tqdm
import seaborn as sns

def tp_fp_fn_analysis(save_folder, text_results_file, xs3d, means3d, ys3d, samples3d, ind_ent_maps):
    # TP TN FN DISTRIBUTION!!
    all_tps = []
    #all_tns = []
    all_fps = []
    all_fns = []

    with torch.no_grad():
        for i in tqdm(range(len(ys3d)), position=0, leave=True, ncols=150):
            samples = samples3d[i]
            mean = means3d[i]
            ent = ind_ent_maps[i].view(-1)
            mean_class = mean.argmax(dim=1).view(-1)
            y = ys3d[i]
            x = xs3d[i].swapaxes(0,1)
            y_flat = y.view(-1)

            tp_loc = torch.where(torch.logical_and(y_flat == 1, mean_class == 1))[0]
            #tn_loc = torch.where(torch.logical_and(torch.logical_and(y_flat == 0, mean_class == 0), x[:,1].reshape(-1) == 1))[0]
            fp_loc = torch.where(torch.logical_and(y_flat == 0, mean_class == 1))[0]
            fn_loc = torch.where(torch.logical_and(torch.logical_and(y_flat == 1, mean_class == 0), x[:,1].reshape(-1) == 1))[0]
            # print(tp_loc.shape)
            # print(ent.view(-1).shape)

            all_tps.append(ent[tp_loc])
            #all_tns.append(ent[tn_loc])
            all_fps.append(ent[fp_loc])
            all_fns.append(ent[fn_loc])

    tps = torch.cat(all_tps)
    #tns = torch.cat(all_tns)
    fps = torch.cat(all_fps)
    fns = torch.cat(all_fns)

    print_and_write(text_results_file, f"tp, fp, fn totals")
    print_and_write(text_results_file, str([tps.shape[0], fps.shape[0], fns.shape[0]]))

    print_and_write(text_results_file, "TP mean", newline=1)
    print_and_write(text_results_file, tps.mean())
    print_and_write(text_results_file, "FP mean", newline=1)
    print_and_write(text_results_file, fps.mean())
    print_and_write(text_results_file, "FN mean", newline=1)
    print_and_write(text_results_file, fns.mean())

    print_and_write(text_results_file, "TP ent values", newline=1)
    print_and_write(text_results_file, tps)
    print_and_write(text_results_file, "FP ent values", newline=1)
    print_and_write(text_results_file, fps)
    print_and_write(text_results_file, "FN ent values", newline=1)
    print_and_write(text_results_file, fns)


    plt.hist(tps, bins=100, color='r');
    plt.ylabel("Voxels per Bin")
    #plt.ylim((0, 350000))
    plt.xlabel("$H$")
    save(save_folder, "tps")

    plt.hist(fps, bins=100, color='r');
    plt.ylabel("Voxels per Bin")
    #plt.ylim((0, 50000))
    plt.xlabel("$H$")
    save(save_folder, "fps")

    plt.hist(fns, bins=100, color='r');
    plt.ylabel("Voxels per Bin")
    #plt.ylim((0, 155000))
    plt.xlabel("$H$")
    save(save_folder, "fns")

    j = -1
    ntps = len(tps)
    nfns = len(fns)
    nfps = len(fps)
    data = {"label":["TP" for _ in range(ntps)][0:j] + ["FN" for _ in range(nfns)][0:j] + ["FP" for _ in range(nfps)][0:j], "ent": torch.cat([tps[0:j], fns[0:j], fps[0:j]]).numpy()}

    plt.figure(figsize=(4, 2.5))
    sns.violinplot(x="label", y="ent", data=data, linewidth=0.5, inner=None)
    plt.ylim((-0.1, 0.8))
    plt.ylabel("$H$")
    save(save_folder, "types_violin")