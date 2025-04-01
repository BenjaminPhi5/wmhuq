import argparse
import SimpleITK as sitk
from wmhuq.preprocessing import load_csv, VPrint, load_image, get_resampled_img, normalize_brain, get_vent_dist, crop_or_pad
from wmhuq.models import load_ssn, load_model_weights
from wmhuq.inference import ssn_ensemble_mean_and_samples, entropy_map_from_samples, reorder_samples, extract_features, load_fazekas_model, load_qc_model
import SimpleITK as sitk
import torch
import pandas as pd
from wmhuq.preprocessing import OUT_SPACING
import os
import math
import time

NUM_SAMPLES = 10
LOW_PERC = 5
HIGH_PERC = 95
SEED = 42

def construct_parser():
    parser = argparse.ArgumentParser(description = "WMH UQ / Fazekas model outputs")
    
    # inputs
    parser.add_argument('-i', type=str, help='path to csv file of input paths', required=True)
    parser.add_argument('--imgfolder', type=str, help='absolute path to image folder', default='.')
    
    # output folder
    parser.add_argument('-o', type=str, help="output folder for outputs", required=True)
    parser.add_argument('-csvname', type=str, help="csv output filename, defaults to wmhuq.csv", default="wmhuq.csv")
    
    # weights folder
    parser.add_argument('-w', type=str, help="path to folder where model weights are stored", required=True)
    
    # options
    parser.add_argument('--resample', action='store_true')
    parser.add_argument('--resize', action='store_true')
    parser.add_argument('--zscore', action='store_true')
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--fazekas', action='store_true')
    parser.add_argument('--features', action='store_true')
    parser.add_argument('--qc', action='store_true')
    parser.add_argument('--saveimgs', action='store_true')
    parser.add_argument('--saveasyougo', action='store_true')
    parser.add_argument('--renormalize', action='store_true')
    
    return parser
    
    
def process_subject(input_files, args, config_models, vprint, fazekas_model=None, qc_model=None):
    
    # load input data
    vprint("loading image data")
    imageid, t1_path, flair_path, mask_path, synthseg_path = input_files
    
    t1_image = sitk.ReadImage(os.path.join(args.imgfolder, t1_path))
    flair_image = sitk.ReadImage(os.path.join(args.imgfolder, flair_path))
    orig_spacing = flair_image.GetSpacing()
    mask_image = sitk.ReadImage(os.path.join(args.imgfolder, mask_path))
    synthseg_image = sitk.ReadImage(os.path.join(args.imgfolder, synthseg_path))
    synthseg_image_orig = sitk.ReadImage(os.path.join(args.imgfolder, synthseg_path))
    
    # calculate ventricle distance map
    vent_dist = get_vent_dist(synthseg_image_orig, args.verbose)
    
    # resample
    if args.resample:
        vprint("resampling")
        t1_image = get_resampled_img(t1_image, OUT_SPACING, original_spacing=None, is_label=False, verbose=args.verbose)
        flair_image = get_resampled_img(flair_image, OUT_SPACING, original_spacing=None, is_label=False, verbose=args.verbose)
        mask_image = get_resampled_img(mask_image, OUT_SPACING, original_spacing=None, is_label=True, verbose=args.verbose)
        synthseg_image = get_resampled_img(synthseg_image, OUT_SPACING, original_spacing=None, is_label=True, verbose=args.verbose)
    
    # get image arrays
    device = args.device
    t1 = torch.from_numpy(sitk.GetArrayFromImage(t1_image)).to(device).type(torch.float32)
    flair = torch.from_numpy(sitk.GetArrayFromImage(flair_image)).to(device).type(torch.float32)
    mask = torch.from_numpy(sitk.GetArrayFromImage(mask_image)).to(device).type(torch.float32)
    synthseg = torch.from_numpy(sitk.GetArrayFromImage(synthseg_image)).to(device).type(torch.float32)
    vent_dist = torch.from_numpy(vent_dist).to(device).type(torch.float32)
    
    # z-score normalize
    if args.zscore:
        vprint("z-score normalizing")
        if args.renormalize:
            t1 = renormalize_brain(t1, synthseg, 't1')
            flair = renormalize_brain(flair, synthseg, 'flair')
        else:
            t1 = normalize_brain(t1, mask, lower_percentile=LOW_PERC, upper_percentile=HIGH_PERC)
            flair = normalize_brain(flair, mask, lower_percentile=LOW_PERC, upper_percentile=HIGH_PERC)
        
    # resize
    orig_dim1, orig_dim2 = flair.shape[1:]
    if args.resize:
        vprint("resizing images")
        t1 = crop_or_pad(t1)
        flair = crop_or_pad(flair)
        mask = crop_or_pad(mask)
        synthseg = crop_or_pad(synthseg)
        vent_dist = crop_or_pad(vent_dist)
    
    # run the model
    vprint("running segmentation model")
    x = torch.stack([flair * mask, t1 * mask, mask])
    mean, samples = ssn_ensemble_mean_and_samples(config_models, x, NUM_SAMPLES, device)
    
    umap = entropy_map_from_samples(samples, normalize=True) / -math.log(0.5)
    p_hat = torch.nn.functional.softmax(mean, dim=1)[:, 1]
    samples = reorder_samples(samples)
    samples = torch.nn.functional.softmax(samples, dim=2)[:,:,1]
    
    # extract model output features
    vprint("extracting UQ features")
    output_feats = extract_features(synthseg, vent_dist, samples, p_hat, umap, args.qc)
    ### TODO COMPLETE: format the features to be inputted to the logreg models.
    # figure out how to turn it into a single row pandas array I think. then the reshape function in the eval code probably needs to go.....

    dfs = []
    # run qc model
    qc_prediction = None
    if args.qc:
        qc_prediction = qc_model(output_feats)
        dfs.append(qc_prediction)
    
    # run fazekas model
    vprint("predicting fazekas and seg. quality")
    if args.fazekas or args.features:
        fazekas_prediction, feature_outputs = fazekas_model(output_feats)
        if args.fazekas:
            dfs.append(fazekas_prediction)
        if args.features:
            dfs.append(feature_outputs)

    # create output dataframe
    df = None
    if len(dfs) > 0:
        df = pd.concat(dfs, axis=1)
        df['ID'] = imageid
        df['wmh_voxel_count'] = (p_hat > 0.5).sum().item()

    # save individual dfs as we go to avoid errors.
    if args.saveasyougo:
        df.set_index('ID', inplace=True)
        csv_path = os.path.join(args.o, f"{args.csvname}_{imageid}")
        df.to_csv(csv_path, index=True)

    # save images if requested.
    if args.saveimgs:
        uq_out_path = os.path.join(args.o, flair_path.split(".nii")[0].split("/")[-1] + "_wmhuqimg.nii.gz")
        seg_out_path = os.path.join(args.o, flair_path.split(".nii")[0].split("/")[-1] + "_wmhuqseg.nii.gz")
        seg = (p_hat > 0.5).type(umap.dtype)
        if args.resize:
            umap = crop_or_pad(umap, w=orig_dim1, d=orig_dim2)
            seg = crop_or_pad(seg, w=orig_dim1, d=orig_dim2)
        umap_img = sitk.GetImageFromArray(umap.cpu().numpy())
        seg_img = sitk.GetImageFromArray(seg.cpu().numpy())
        umap_img.SetSpacing(flair_image.GetSpacing())
        seg_img.SetSpacing(flair_image.GetSpacing())
        if args.resample:
            umap_img = get_resampled_img(umap_img, orig_spacing, is_label=False, verbose=args.verbose)
            seg_img = get_resampled_img(seg_img, orig_spacing, is_label=True, verbose=args.verbose)
        sitk.WriteImage(umap_img, uq_out_path)
        sitk.WriteImage(seg_img, seg_out_path)
    
    return df
    
def main(args):
    
    # get paths to input files
    inputs = pd.read_csv(args.i, header=None)
    column_names = 'ID t1 flair mask synthseg'.split(' ')
    inputs.columns = column_names
    N = inputs.shape[0]
        
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
    if args.gpu:
        config_models = [m.cuda().eval() for m in config_models]
    
    vprint("loading fazekas and qc model weights")
    fazekas_model = load_fazekas_model(args.w)
    qc_model = load_qc_model(args.w)
        
    # run subject loop
    results = []
    for idx, row in inputs.iterrows():
        print(f"# processing image-set: {idx+1}/{N} : {row[0]}")
        start = time.time()
        try:
            result = process_subject(row, args, config_models, vprint, fazekas_model, qc_model)
            results.append(result)
            
        except Exception as e:
            print(f"failed for image id: {row[0]}")
            print(e)
        end = time.time()
        print(f"time: {end - start}")

    combined_df = None
    if args.fazekas or args.qc or args.features:
        combined_df = pd.concat(results, axis=0)
        if not args.saveasyougo:
            combined_df.set_index('ID', inplace=True)
        csv_path = os.path.join(args.o, args.csvname)
        combined_df.to_csv(csv_path, index=True)
    

if __name__ == '__main__':
    torch.manual_seed(SEED)
    parser = construct_parser()
    args = parser.parse_args()
    main(args)
