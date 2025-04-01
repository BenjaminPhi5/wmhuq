import SimpleITK as sitk 
import os
import argparse
from tqdm import tqdm

def check_image_orientation(filepath):
    """
    set the image to 'RAS' orientation
    """
    img = sitk.ReadImage(filepath)
    img = sitk.DICOMOrient(img, 'RAS') # RAS orientation is the orientation used by the CVD and MSS3 data
    sitk.WriteImage(img, filepath)

def reorient_all_images_in_folder(folder):
    nifti_files = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith(".nii.gz"):
                nifti_files.append(os.path.join(root, file))
    
    print("num files found: ", len(nifti_files))
    for f in tqdm(nifti_files):
        check_image_orientation(f)


def construct_parser():
    # preprocessing settings
    parser = argparse.ArgumentParser(description = "orient all .nii.gz images in folder (recursive to subfolders) to RAS DICOM orientation")
    
    parser.add_argument('-i', '--in_dir', required=True, help="folder path", type=str)
    
    return parser

if __name__ == '__main__':
    parser = construct_parser()
    args = parser.parse_args()
    print(args.in_dir)
    reorient_all_images_in_folder(args.in_dir)
