import matplotlib.pyplot as plt
import os

def save(save_dir, fname, small=False, is_img=False):
    """
    for saving the currently plotted object
    """
    plt.tight_layout()
    if small:
        plt.figure(figsize=(4,3))
    if is_img:    
        fname = "images/" + fname
    plt.savefig(os.path.join(save_dir, fname), bbox_inches = "tight")
    plt.show()
    plt.clf()
    plt.close()
    
def imsave(save_dir, fname, image, cmap, title="", small=False, is_img=True, vmin=None, vmax=None, show=True, origin='upper'):
    """
    for saving an image (that would be otherwise shown in notebook with plt.imshow
    """
    plt.tight_layout()
    if small:
        plt.figure(figsize=(4,3))
    if is_img:    
        fname = "images/" + fname
    if show:
        # plt.axis('off')
        # plt.title(title)
        # plt.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
        # plt.clf()
        # plt.close()
        pass
        
    plt.axis('off')
    plt.title(title)
    plt.imsave(os.path.join(save_dir, fname + ".png"), image, cmap=cmap, format="png", vmin=vmin, vmax=vmax, origin=origin)
    plt.clf()
    plt.close()