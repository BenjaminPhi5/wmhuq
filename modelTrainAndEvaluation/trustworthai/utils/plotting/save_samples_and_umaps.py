import matplotlib.pyplot as plt
from trustworthai.utils.uncertainty_maps.entropy_map import entropy_map_from_samples
from trustworthai.utils.plotting.saving_plots import save, imsave
import os

def plot_and_save_samples_and_umaps(save_folder, samples3d, ys3d, ind_ent_maps, scan_ids, slice_ids):
    """
    samples3d, a list of samples of 3D brain scan outputs from a stochastic
    model
    
    save_folder: folder to save the images to
    scan_ids: the ids of the scans to plot
    slice_ids: for each entry in scan_ids, the target slice of interest.
    scan_ids can be repeated to plot more than one slice from that slice
    
    assumes there are max 20 samples
    """
    assert len(samples3d[0]) <= 20
    
    for s in range(len(scan_ids)):

        scan_id = scan_ids[s]
        scan_ent_map = ind_ent_maps[s]

        slice_id = slice_ids[s]
        count = 0
        samples = samples3d[scan_id][:,slice_id]
        plt.figure(figsize=(30,8))
        for i in range(2*10):
            plt.subplot(2, 10, count+1)
            plt.imshow(samples[i].argmax(dim=0), cmap='gray')
            plt.title(i)
            plt.axis('off')
            count += 1
        save(save_folder, f"all_samples- {scan_id}-{slice_id}", is_img=True)

        for i in range(len(samples)):
            # plt.imshow(samples[i].argmax(dim=0), cmap='gray')
            imsave(save_folder, f"sample- {scan_id}-{slice_id}-{i}",
                   samples[i].argmax(dim=0),
                   cmap='gray',
                   title=i, is_img=True, small=True, show=False)

        # show ground truth and uncertainty map
        slice_ent_map = scan_ent_map[slice_id]
        imsave(save_folder, f"GT- {scan_id}-{slice_id}-{i}", ys3d[scan_id][slice_id], cmap='gray', title=i, is_img=True, small=True)
        imsave(save_folder, f"ent_map- {scan_id}-{slice_id}", slice_ent_map, cmap=None, title=i, is_img=True, small=True, vmin=0, vmax=0.7)