import subprocess
import os
from tqdm import tqdm

path = "/home/s2208943/datasets/Inter_observer/"

folders = os.listdir(path)

for d in tqdm(folders):
    try:
        files = os.listdir(os.path.join(path, d))
    except:
        print(f"{d} does not appear to be a folder")
        continue
    for f in files:
        if ".nii" in f and ".gz" not in f:
            print(f"called for {f}")
            print(os.path.join(path, d, f),
                os.path.join(path, d, f + ".gz"))
            os.system(
                f"gzip {os.path.join(path, d, f)} {os.path.join(path, d, f)}.gz"
            )