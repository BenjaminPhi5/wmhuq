import sys
import os
import argparse
import subprocess

# TODO sort out these imports so that they work like a proper module should
from twaidata.MRI_preprep.io import load_nii_img, save_nii_img, FILETYPE
from twaidata.MRI_preprep.normalize_brain import normalize_brain
from twaidata.MRI_preprep.resample import resample_and_return, resample_and_save
from twaidata.mri_dataset_directory_parsers.parser_selector import select_parser

def construct_parser():
    # preprocessing settings
    parser = argparse.ArgumentParser(description = "MRI nii.gz simple preprocessing pipeline")
    
    parser.add_argument('-i', '--in_dir', required=True, help='Path to parent of the dataset to be preprocessed')
    parser.add_argument('-o', '--out_dir', required=True, help='Path to the preprocessed data folder')
    parser.add_argument('-n', '--name', required=True, help='Name of dataset to be processed')
    parser.add_argument('-s', '--start', default=0, type=int, help='individual in dataset to start from (if start = 0 will start from the first person in the dataset, if 10 will start from the 11th)')
    parser.add_argument('-e', '--end', default=-1, type=int, help="individual in dataset to stop at (if end = 10 will end at the 10th person in the dataset)")
    parser.add_argument('--out_spacing', default="1.,1.,3.", type=str, help="output spacing used in the resampling process. Provide as a string of comma separated floats, no spaces, i.e '1.,1.,3.")

    return parser


def main(args):
    # get the file selection map
    parser = select_parser(args.name, args.in_dir, args.out_dir)
    
    # get the files to be processed
    files_map = parser.get_dataset_inout_map()
    keys = sorted(list(files_map.keys()))
    
    # select range of files to preprocess
    if args.end == -1:
        keys = keys[args.start:]
    else:
        keys = keys[args.start:args.end]
        
    print(f"starting at individual {args.start} and ending at individual {args.end}")
    
    # get the fsl directory
    FSLDIR = os.getenv('FSLDIR')
    if 'FSLDIR' == "":
        raise ValueError("FSL is not installed. Install FSL to complete brain extraction")
        
        
    # parse the outspacing argument
    outspacing = [float(x) for x in args.out_spacing.split(",")]
    if len(outspacing) != 3: # 3D
        raise ValueError(f"malformed outspacing parameter: {args.out_spacing}")
    else:
        print(f"using out_spacing: {outspacing}")
    
    # run preprocessing
    for key in keys:
        print(f"processing: {key}")
        output_dir = files_map[key]['outpath']
        output_filename = files_map[key]['outfilename']
        islabel = files_map[key]['islabel']
        
        # check that the file exists
        if not os.path.exists(key):
            raise ValueError(f"target file doesn't exist: {key}")
        # create the output directory if it does not exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            

        # bias field correction (for T1 only...)
        # 
        
        # BRAIN EXTRACTION
        if not islabel:
            # do the brain extraction - running bet2 from the fsl library
            bet_file = f"BET_{output_filename}" # fsl adds on the .nii.gz automatically
            bet_command = [os.path.join(*[FSLDIR,'bin', 'bet2']), key, os.path.join(output_dir, bet_file)]
            _ = subprocess.call(bet_command)
            bet_filepath = os.path.join(output_dir, bet_file + FILETYPE) # because fsl added on .nii.gz in above step.
            next_filepath = bet_filepath
        else:
            next_filepath = key # label images don't need brain extracting
            
            
        # NORMALIZE
        # label images don't need normalizing
        if not islabel:
            # do the normalizing
            img, header = load_nii_img(next_filepath)
            normalize_brain(img) # in place operation

            # save the results
            normalize_filepath = f"NORMALIZE_{output_filename}{FILETYPE}"
            save_nii_img(os.path.join(output_dir, normalize_filepath), img, header)
            next_filepath = normalize_filepath
    
        # RESAMPLE
        # do the resampling
        output_file = os.path.join(output_dir, output_filename + FILETYPE)
        next_file = os.path.join(output_dir, next_filepath)
        resample_and_save(next_file, output_file, is_label=islabel, outspacing=outspacing)
        
        print()
        
if __name__ == '__main__':
    parser = construct_parser()
    args = parser.parse_args()
    main(args)