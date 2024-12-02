# Uncertainty quantification for White Matter Hyperintensity segmentation detects silent failures and improves automated Fazekas quantification

### paper link:
Please find the preprint on arxiv: [https://arxiv.org/pdf/2411.17571]

### training code:
information coming soon

### running inference
First acquire the model weights data from (link coming soon). To run the script, use the following command:

example command:

```
python wmhuq/main.py -i <PATH TO CSVs>/test2_csv.txt -o <OUTPUT FOLDER> -w <WEIGHT FOLDER> --imgfolder <IMAGE FOLDER> --resample --resize --zscore --gpu --saveasyougo --saveimg --qc --fazekas --features 
```

unzip the files before starting.

Note below that specifying the --saveimgs and --qc script will make the script slower. You may wish to exclude these flags and just leave --fazekas and --features.

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
  - -i: path to a csv file containing image paths. each row is for a separate subject. each row should be of the form: image id, t1 path, flair path, mask path, synthseg path. These paths can be relative paths if you also specify --imgfolder
  - --imgfolder: absolute path that is prepended to the relative paths in the input csv file (-i)
    
  - -o the desired output folder for derived results.
  - 
  - --resample (resample images internally to 1x1x3 space)
  - --zscore (perform z-score normalization to images)
  - --gpu perform all processing (where possible) on the gpu
  - --verbose
  - --fazekas (compute fazekas score)
  - --features (output features used to compute fazekas score)
  - --qc (compute segmentation quality score)
  - --saveimgs (save the UQ map outputs)
  - --saveasyougo save the results of each individual to a csv file as they are processed. useful if processing a large number of subjects and the script crashes without recovery for some reason.


output (all are optional):
  - wmh UQ map (specify --saveimgs)
  - fazekas probabilistic score (specify --fazekas)
  - segmentation quality score (specify --qc)
  - input features for fazekas score (specify --features)
