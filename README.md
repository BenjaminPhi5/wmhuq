# Uncertainty quantification for White Matter Hyperintensity segmentation detects silent failures and improves automated Fazekas quantification

### inference using trained models

clone this repository
```
cd wmhuq-eval
conda create -y --name wmhuq python=3.10
conda activate wmhuq
pip install -e .
```


input: flair, t1, brain mask and synthseg output, as .nii.gz images
requirements: 
  - both t1 and flair have been skull stripped and bias corrected.
  - all images are in the same coregistered space

flags:
  - -t1 (required) T1w image
  - -flair (required) FLAIR image
  - -mask (required) brain mask image
  - -synthseg (required) synthseg output image
  - -models (required) path to the folder where the model weights are stored.
  - -o (required) output folder path
  - --resample (resample images internally to 1x1x3 space)
  - --zscore (perform z-score normalization to images)
  - --gpu (perform all processing (where possible) on the gpu
  - --verbose
  - --fazekas (compute fazekas score)
  - --qc (compute segmentation quality score)
  - --wmhparc (compute parcellation of the wmh segmentation)
  - --delimages (delete image outputs)

processing (some optional depending on flags):
  - resampling of all images to 1x1x3 space
  - z-score normalization of t1 and flair
  - passing images through the model

output:
  - wmh segmentation
  - wmh UQ map
  - fazekas probabilistic score
  - segmentation quality score
  - input features for fazekas score
