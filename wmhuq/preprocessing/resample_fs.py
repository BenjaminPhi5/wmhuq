import subprocess

def resample(image_path, out_path, voxel_sizes, use_nearest_neighbor=False):
    rt = "interpolate"
    if use_nearest_neighbor:
        rt = "nearest"
    
    voxel_sizes = [str(v) for v in voxel_sizes]
    command = [
        "mri_convert", "-vs", *voxel_sizes, "-rt", rt,
        image_path, out_path
    ]
    
    _ = subprocess.call(command, stdout=subprocess.DEVNULL)
    
    return out_path