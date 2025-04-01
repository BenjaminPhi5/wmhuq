import matplotlib.pyplot as plt
from typing import Dict, Tuple
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import numpy as np
import kornia as K
from trustworthai.utils.plotting.saving_plots import save
from trustworthai.utils.print_and_write_func import print_and_write
from tqdm import tqdm

def create_random_labels_map(classes: int) -> Dict[int, Tuple[int, int, int]]:
    labels_map: Dict[int, Tuple[int, int, int]] = {}
    for i in classes:
        labels_map[i] = torch.randint(0, 255, (3, ))
    labels_map[0] = torch.zeros(3)
    return labels_map

def labels_to_image(img_labels: torch.Tensor, labels_map: Dict[int, Tuple[int, int, int]]) -> torch.Tensor:
    """Function that given an image with labels ids and their pixels intrensity mapping, creates a RGB
    representation for visualisation purposes."""
    assert len(img_labels.shape) == 2, img_labels.shape
    H, W = img_labels.shape
    out = torch.empty(3, H, W, dtype=torch.uint8)
    for label_id, label_val in labels_map.items():
        mask = (img_labels == label_id)
        for i in range(3):
            out[i].masked_fill_(mask, label_val[i])
    return out

def show_components(img, labels):
    color_ids = torch.unique(labels)
    labels_map = create_random_labels_map(color_ids)
    labels_img = labels_to_image(labels, labels_map)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,12))

    # Showing Original Image
    ax1.imshow(img)
    ax1.axis("off")
    ax1.set_title("Orginal Image")

    #Showing Image after Component Labeling
    ax2.imshow(labels_img.permute(1,2,0).squeeze().numpy())
    ax2.axis('off')
    ax2.set_title("Component Labeling")
    

def conn_comp_2d_analysis(save_folder, results_file, uncertainty_thresholds, ys3d, means3d, ind_ent_maps):
    conncomp_outs = []

    for y in tqdm(ys3d, position=0, leave=True, ncols=150):
        labels_out = K.contrib.connected_components(y.unsqueeze(1).type(torch.float32), num_iterations=150)
        conncomp_outs.append(labels_out)

    # this is the 1 pixel is covered by the entropy
    #c_thresholds = [0.05, 0.1, 0.2, 0.3, 0.6]
    c_thresholds = [t.item() for t in uncertainty_thresholds]
    coverages = [0.1, 0.5, 0.9]
    missing_lesion_size_ent = []
    existing_lesion_size_ent = []

    missing_lesion_size_mean = []

    num_entirely_missed_lesions = {ct:0 for ct in c_thresholds}
    entirely_missed_lesions_size = {ct:[] for ct in c_thresholds}
    proportion_missing_lesion_covered_ent = {ct:[] for ct in c_thresholds}
    num_lesions = 0
    sizes = []
    missing_area_sizes = []
    missing_area_coverage = {ct:[] for ct in c_thresholds}
    for batch in tqdm(range(len(ys3d)), position=0, leave=True, ncols=150):
        for i in range(0, ys3d[batch].shape[0], 3):
            conncomps = conncomp_outs[batch][i]
            ent = ind_ent_maps[batch][i]
            mean = means3d[batch].argmax(dim=1)[i]

            ids = conncomps.unique()[1:]
            for idx in ids:
                cc = (conncomps == idx)
                num_lesions += 1
                size = torch.sum(cc)
                sizes.append(size)

                missing_area = (mean == 0) & cc
                ma_size = missing_area.sum()
                missing_area_sizes.append(ma_size)

                # get uncertain pixels for each threshold
                for tau in c_thresholds:
                    uncert = (ent > tau).type(torch.long)

                    # coverage proportion
                    coverage = (uncert * missing_area).sum() / ma_size
                    missing_area_coverage[tau].append(coverage)


                    if torch.max(mean * cc) == 0:
                        # proportion of those lesions that are missing from mean covered by uncertainty
                        proportion_missing_lesion_covered_ent[tau].append(torch.sum(uncert * cc) / size)

                        # lesions entirely missed by both mean prediction and uncertainty map
                        # i.e not a single voxel is identified as uncertain or mean, total silent failure.
                        if torch.max(uncert * cc) == 0:
                            num_entirely_missed_lesions[tau] += 1
                            entirely_missed_lesions_size[tau].append(size)

    # replace nans and convert to tensor
    for tau in c_thresholds:
        missing_area_coverage[tau] = torch.Tensor([c.item() if not torch.isnan(c) else 0 for c in missing_area_coverage[tau]])

    print_and_write(results_file, "uncertainty thresholds", newline=1)
    print_and_write(results_file, uncertainty_thresholds)

    print_and_write(results_file, "mean coverage of areas missed by mean as tau increases", newline=1)
    # for tau in c_thresholds:
    #     print_and_write(results_file, f"{tau}: {missing_area_coverage[tau].mean().item()}", newline=1)
    # print_and_write(results_file, "",newline=1)
    print_and_write(results_file, torch.Tensor([missing_area_coverage[tau].mean().item() for tau in c_thresholds]))

    print_and_write(results_file, "mean size of entirely missed lesions", newline=1)
    # for tau in c_thresholds:
    #     print_and_write(results_file, f"{tau}: {torch.Tensor(entirely_missed_lesions_size[tau]).mean().item()}", newline=1)
    # print_and_write(results_file, "",newline=1)
    print_and_write(results_file, torch.Tensor([torch.Tensor(entirely_missed_lesions_size[tau]).mean().item() for tau in c_thresholds]))


    print_and_write(results_file, "mean coverage of lesions entirely missed by the mean segmentation", newline=1)
    # for tau in c_thresholds:
    #     print_and_write(results_file, f"{tau}: {torch.Tensor(proportion_missing_lesion_covered_ent[tau]).mean().item()}", newline=1)
    # print_and_write(results_file, "",newline=1)
    print_and_write(results_file, torch.Tensor([torch.Tensor(proportion_missing_lesion_covered_ent[tau]).mean().item() for tau in c_thresholds]))

    print_and_write(results_file, "total number of missing lesions", newline=1)
    # for tau in c_thresholds:
    #     print_and_write(results_file, f"{tau}: {num_entirely_missed_lesions[tau]}", newline=1)
    # print_and_write(results_file, "",newline=1)
    print_and_write(results_file, torch.Tensor([num_entirely_missed_lesions[tau] for tau in c_thresholds]))



    print_and_write(results_file, "proportion of lesions entirely missed", newline=1)
    # for tau in c_thresholds:
    #     print_and_write(results_file, f"{tau}: {num_entirely_missed_lesions[tau]/num_lesions}", newline=1)
    # print_and_write(results_file, "",newline=1)
    print_and_write(results_file, torch.Tensor([num_entirely_missed_lesions[tau]/num_lesions for tau in c_thresholds]))



    # print("total number of missing lesions")
    # print([(tau, num) for tau, num in num_entirely_missed_lesions.items()])
    # print("proportion of lesions entirely missed")
    # print([(tau, num/num_lesions) for tau, num in num_entirely_missed_lesions.items()])

    print_and_write(results_file, f"num lesions:", newline=1)
    print_and_write(results_file, num_lesions)