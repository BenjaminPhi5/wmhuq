import os

def scratch_dir():
    # determine scratch_dir
    if os.path.exists("/disk/scratch_big/"):
        scratch_dir = "/disk/scratch_big/s2208943/ipdis/"
    elif os.path.exists("/disk/scratch/"):
        scratch_dir = "/disk/scratch/s2208943/ipdis/"
    elif os.path.exists("/media/benp/NVMEspare/datasets/"):
        scratch_dir = "/media/benp/NVMEspare/datasets/"
        print("running on local machine!")
    else:
        raise ValueError("scratch disk for placing results not found")
    return scratch_dir