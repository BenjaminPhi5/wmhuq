import argparse
from wmhuq.preprocessing import load_csvs, VPrint, load_image, get_resampled_img, normalize_brain, get_vent_dist, crop_or_pad
from wmhuq.models import load_ssn, load_model_weights
from wmhuq.inference import ssn_ensemble_mean_and_samples, entropy_map_from_samples, reorder_samples, extract_features, load_fazekas_model, load_qc_model
import SimpleITK as sitk
import torch
import pandas as pd
from wmhuq.preprocessing import OUT_SPACING
import os
import math

NUM_SAMPLES = 10
LOW_PERC = 5
HIGH_PERC = 95
SEED = 42

def construct_parser():
    parser = argparse.ArgumentParser(description = "WMH UQ / Fazekas model outputs")
    
    # inputs
    parser.add_argument('-t1', type=str, help='T1w image path or path to csv file containing list of T1w images', required=True)
    parser.add_argument('-flair', type=str, help='FLAIR image path or path to csv file containing list of FLAIR images', required=True)
    parser.add_argument('-mask', type=str, help='brain mask image path or path to csv file containing list of brain mask images', required=True)
    parser.add_argument('-synthseg', type=str, help='synthseg aseg image path or path to csv file containing list of synthseg seg images', required=True)

    
    # output folder
    parser.add_argument('-o', type=str, help="output folder path or match. If set to match, outputs to the same folder as the flair image", required=True)
    
    # weights folder
    parser.add_argument('-w', type=str, help="path to folder where model weights are stored", required=True)
    
    # options
    parser.add_argument('--resample', action='store_true')
    parser.add_argument('--resize', action='store_true')
    parser.add_argument('--zscore', action='store_true')
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--fazekas', action='store_true')
    parser.add_argument('--qc', action='store_true')
    parser.add_argument('--saveimgs', action='store_true')
    
    return parser
    
    
def process_subject(input_files, args, config_models, vprint, fazekas_model=None, qc_model=None):
    
    # load input data
    vprint("loading image data")
    t1_path, flair_path, mask_path, synthseg_path = input_files
    t1_image = sitk.ReadImage(t1_path)
    flair_image = sitk.ReadImage(flair_path)
    mask_image = sitk.ReadImage(mask_path)
    synthseg_image = sitk.ReadImage(synthseg_path)
    synthseg_image_orig = sitk.ReadImage(synthseg_path)
    
    # calculate ventricle distance map
    vent_dist = get_vent_dist(synthseg_image_orig, args.verbose)
    
    # resample
    if args.resample:
        vprint("resampling")
        t1_image = get_resampled_img(t1_image, OUT_SPACING, is_label=False, verbose=args.verbose)
        flair_image = get_resampled_img(flair_image, OUT_SPACING, is_label=False, verbose=args.verbose)
        mask_image = get_resampled_img(mask_image, OUT_SPACING, is_label=True, verbose=args.verbose)
        synthseg_image = get_resampled_img(synthseg_image, OUT_SPACING, is_label=True, verbose=args.verbose)
    
    # get image arrays
    device = args.device
    t1 = torch.from_numpy(sitk.GetArrayFromImage(t1_image)).to(device)
    flair = torch.from_numpy(sitk.GetArrayFromImage(flair_image)).to(device)
    mask = torch.from_numpy(sitk.GetArrayFromImage(mask_image)).to(device)
    synthseg = torch.from_numpy(sitk.GetArrayFromImage(synthseg_image)).to(device)
    vent_dist = torch.from_numpy(vent_dist).to(device)
    
    print("input shapes t1 flair mask synthseg")
    print(t1.shape, flair.shape, mask.shape, synthseg.shape)
    
    # z-score normalize
    if args.zscore:
        vprint("z-score normalizing")
        t1 = normalize_brain(t1, mask, lower_percentile=LOW_PERC, upper_percentile=HIGH_PERC)
        flair = normalize_brain(flair, mask, lower_percentile=LOW_PERC, upper_percentile=HIGH_PERC)
        
    # resize
    if args.resize:
        vprint("resizing images")
        t1 = crop_or_pad(t1)
        flair = crop_or_pad(flair)
        mask = crop_or_pad(mask)
        synthseg = crop_or_pad(synthseg)
        vent_dist = crop_or_pad(vent_dist)
    
    # run the model
    vprint("running segmentation model")
    x = torch.stack([flair, t1, mask])
    mean, samples = ssn_ensemble_mean_and_samples(config_models, x, NUM_SAMPLES, device)
    umap = entropy_map_from_samples(samples, normalize=True) / -math.log(0.5)
    p_hat = torch.nn.functional.softmax(mean, dim=1)[:, 1]
    samples = reorder_samples(samples)
    samples = torch.nn.functional.softmax(samples, dim=2)[:,:,1]
    
    print("SIZES")
    print("samples: ", samples.shape)
    print("mean: ", mean.shape)
    print("umap: ", umap.shape)
    print("vent dist: ", vent_dist.shape)
    
    # extract model output features
    vprint("extracting UQ features")
    output_feats = extract_features(synthseg, vent_dist, samples, p_hat, umap)
    ### TODO COMPLETE: format the features to be inputted to the logreg models.
    # figure out how to turn it into a single row pandas array I think. then the reshape function in the eval code probably needs to go.....
    
    def print_keys(m):
        if isinstance(m, dict):
            for key in m.keys():
                print("key: ", key)
                print_keys(m[key])
    
    print_keys(output_feats)
            
    
    # run fazekas model
    vprint("predicting fazekas and seg. quality")
    fazekas_prediction = fazekas_model(output_feats)
    
    # run qc model
    qc_prediction = qc_model(output_feats)
    
    # store model outputs
    # create a pandas csv here?
    print(fazekas_prediction)
    print("---------")
    print(qc_prediction)
    
    
def main(args):
    
    # get paths to input files
    is_csv = args.t1.endswith(".csv")
    input_files = args.t1, args.flair, args.mask, args.synthseg
    if is_csv:
        t1s, flairs, masks, synthsegs = load_csvs(input_files)
    else:
        t1s, flairs, masks, synthsegs = input_files
        t1s, flairs, masks, synthsegs = [t1s], [flairs], [masks], [synthsegs]
        
    # verbose?
    vprint = VPrint(args.verbose)
    
    # load model and model weights
    vprint("Loading segmentation model weights")
    args.device = "cuda" if args.gpu else "cpu"
    model = load_ssn(args.device)
    model = model.eval()
    weight_names = [f"ssn_ens{i}_cv0" for i in range(10)]
    weight_paths = [os.path.join(args.w, wn) for wn in weight_names]
    for wp in weight_paths:
        if not os.path.exists(wp):
            raise ValueError("model weights not found")
    config_models = [load_model_weights(model, wp) for wp in weight_paths]
    
    vprint("loading fazekas and qc model weights")
    fazekas_model = load_fazekas_model(args.w)
    qc_model = load_qc_model(args.w)
        
    # run subject loop
    for idx, inputs in enumerate(zip(t1s, flairs, masks, synthsegs)):
        vprint(f"processing image-set: {idx}")
        process_subject(inputs, args, config_models, vprint, fazekas_model, qc_model)
    

if __name__ == '__main__':
    torch.manual_seed(SEED)
    parser = construct_parser()
    args = parser.parse_args()
    main(args)
