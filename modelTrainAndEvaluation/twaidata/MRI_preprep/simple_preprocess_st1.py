import sys
import os
import argparse
from natsort import natsorted
from abc import ABC, abstractmethod
import subprocess

# todo sort out these imports so that they work like a proper module should.
from twaidata.MRI_preprep.io import load_nii_img, save_nii_img, FORMAT
from twaidata.MRI_preprep.normalize_brain import normalize_brain
from twaidata.MRI_preprep.resample import resample_and_return, resample_and_save
from twaidata.mri_dataset_directory_parsers.parser_selector import select_parser


def construct_parser():
    # preprocessing settings
    parser = argparse.ArgumentParser(description = "MRI nii.gz simple preprocessing pipeline")
    
    parser.add_argument('-i', '--in_dir', required=True, help='Path to parent of the dataset to be preprocessed')
    parser.add_argument('-o', '--out_dir', required=True, help='Path to the preprocessed data folder')
    parser.add_argument('-c', '--csv_file', default=None, help='CSV file containing preprocessing data for custom datasets')
    parser.add_argument('-n', '--name', required=True, help='Name of dataset to be processed')
    parser.add_argument('-s', '--start', default=0, type=int, help='individual in dataset to start from (if start = 0 will start from the first person in the dataset, if 10 will start from the 11th)')
    parser.add_argument('-e', '--end', default=-1, type=int, help="individual in dataset to stop at (if end = 10 will end at the 10th person in the dataset)")
    parser.add_argument('--out_spacing', default="1.,1.,3.", type=str, help="output spacing used in the resampling process. Provide as a string of comma separated floats, no spaces, i.e '1.,1.,3.")
    parser.add_argument('-f', '--force_replace', default="False", type=str, help="if true, files that already exist in their target preproessed form will be overwritten (set to true if a new preprocessing protocol is devised, otherwise leave false for efficiency)")
    parser.add_argument('-z', '--skip_if_any', default="False", type=str, help="if true, skips an individual if a single post processed file for that individual is found (useful when running across multiple machines to same output folder")
    parser.add_argument('-a', '--add_dsname_to_folder_name', default="True", type=str)

    return parser


def main(args):
    
    # ======================================================================================
    # SETUP PREPROCESSING PIPELINE
    # ======================================================================================
    
    # get the parser that maps inputs to outputs
    # csv file used for custom datasets
    parser = select_parser(args.name, args.in_dir, args.out_dir, args.csv_file, args.add_dsname_to_folder_name.lower() == "true")
    
    # get the files to be processed
    files_map = parser.get_dataset_inout_map()
    keys = natsorted(list(files_map.keys()))
    
    # select range of files to preprocess
    if args.end == -1:
        keys = keys[args.start:]
    else:
        keys = keys[args.start:args.end]
        
    print(f"starting at individual {args.start} and ending at individual {args.end}")
    
    # get the fsl directory used for brain extraction and bias field correction
    FSLDIR = os.getenv('FSLDIR')
    if 'FSLDIR' == "":
        raise ValueError("FSL is not installed. Install FSL to complete brain extraction")
        
    # parse the outspacing argument
    outspacing = [float(x) for x in args.out_spacing.split(",")]
    if len(outspacing) != 3: # 3D
        raise ValueError(f"malformed outspacing parameter: {args.out_spacing}")
    else:
        print(f"using out_spacing: {outspacing}")
            
    
    # ======================================================================================
    # RUN
    # ======================================================================================
    for ind in keys:
        any_found = False
        print(f"processing individual: {ind}")
        ind_filemap = files_map[ind]
        
        # check whether all individuals have been done and can therefore be skipped
        can_skip = True
        if not args.force_replace.lower() == "true":
            for filetype in files_map[ind].keys():
                output_dir = ind_filemap[filetype]['outpath']
                output_filename = ind_filemap[filetype]['outfilename']

                if output_filename != None and not os.path.exists(os.path.join(output_dir, output_filename + FORMAT)):
                    can_skip = False
                    break
                else:
                    any_found = True
        else:
            can_skip = False

        # a temp file created to tell other threads that this individual is being processed
        # used when the z flag is true.
        tmp_file = os.path.join(output_dir, f"{ind}_temp.txt")
        if os.path.exists(tmp_file):
            any_found = True
            
        can_skip = can_skip or (any_found and args.skip_if_any.lower() == "true")
        if can_skip:
            print(f"skipping, because preprocessed individual {ind} file exists and force_replace set to false")
            continue
        
        # create the output directory if it does not exist
        if not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
                
        # setting the temp file
        with open(tmp_file, "w") as f:
            f.write(f"processing {ind}")
        
        for filetype in natsorted(files_map[ind].keys()):
            if filetype == "ICV":
                continue
                
            print(f"processing filetype: ", filetype)
            
            infile = ind_filemap[filetype]['infile']
            output_dir = ind_filemap[filetype]['outpath']
            output_filename = ind_filemap[filetype]['outfilename']
            islabel = ind_filemap[filetype]['islabel']
            
            print(f"processing file: {infile}")
        
            # check that the file exists
            if not os.path.exists(infile):
                raise ValueError(f"target file doesn't exist: {infile}")
            
            next_file = infile.split(FORMAT)[0]
            # ======================================================================================
            # BIAS FIELD CORRECTION (T1 ONLY HERE)
            # ======================================================================================

            # only applied to T1 (the ED data doesn't need it for flair and I'm not sure about WMH challenge...)
            if filetype == "T1":
                # define name of file to be saved
                out_file = os.path.join(output_dir, f"{output_filename}_BIAS_CORR")
                
                # fast outputs many files to the original folder the data is located in.
                # The relvant file (_restore.nii.gz) is copied over.
                bias_field_corr_command = [os.path.join(*[FSLDIR,'bin', 'fast']), '-b', '-B', next_file + FORMAT]
                _ = subprocess.call(bias_field_corr_command)
                
                corrected_file = next_file.split(".nii.gz")[0] + "_restore.nii.gz"
                _ = subprocess.call(["cp", corrected_file, out_file + FORMAT])
            
                next_file = out_file
                print("outfile post bfc: ", out_file)
        
            # ======================================================================================
            # BRAIN EXTRACTION
            # ======================================================================================
            if not islabel:
                out_file = os.path.join(output_dir, f"{output_filename}_BET")
                flair_outdir = ind_filemap["FLAIR"]["outpath"]
                flair_outfilename = ind_filemap["FLAIR"]["outfilename"]
                mask_out_file = os.path.join(flair_outdir, flair_outfilename + "_BET_mask")
                
                # check to see if the file has a ICV volume file defined
                # if so, use it as the brain extraction mask, otherwise, run BET
                if "ICV" in ind_filemap and os.path.exists(ind_filemap["ICV"]["infile"]):
                    # multiply ICV mask by brain mask
                    icv_filepath = ind_filemap["ICV"]["infile"]
                    icv, _ = load_nii_img(icv_filepath)
                    img, header = load_nii_img(next_file)
                    img = img.squeeze()
                    
                    img = img * icv.squeeze()
                    save_nii_img(out_file, img, header)
                    
                    # copy the ICV to the output directory
                    if filetype == "FLAIR":
                        cp_command = ['cp', icv_filepath, mask_out_file + FORMAT]
                        _ = subprocess.call(cp_command)
                    
                elif filetype != "FLAIR":
                    # if it isn't a flair file, use the processed BET flair file as a map
                    # load the BET processed flair file (flairs must be processced first):
                    bet_flair, _ = load_nii_img(mask_out_file)
                    
                    # load the target image (say T1)
                    img, header = load_nii_img(next_file)
                    img = img.squeeze()
                    
                    # apply the mask
                    img = img * bet_flair
                    save_nii_img(out_file, img, header)
                    
                else:
                    # if image type is flair, generate the bet mask
                    # flair images must be run first so that the flair bet mask exists when preprocessing the t1
                    # run BET tool
                    # bet outputs the result and the mask
                    
                    # if there should have been an icv file use bet with -S for a better extraction, otherwise just use the simpler preprocessed
                    # version for now (ideally every file will be preprocessed with the -S flag but it is a lot slower)
                    # if "ICV" in ind_filemap: # i.e others in the datset have ICV e.g for the Ed Data but for this individual the ICV file couldn't be found.
                    #     bet_command = [os.path.join(*[FSLDIR,'bin', 'bet']), next_file + FORMAT, out_file, "-m", "-S"]   
                    # else:
                    #     bet_command = [os.path.join(*[FSLDIR,'bin', 'bet2']), next_file + FORMAT, out_file, "-m"]
                    
                    # UPDATE: I have just switched to using the more robust bet2 with -S option at all times for the ADNI data.
                    # (bet is bet2 + options..wierd...)
                    # print(FSLDIR, next_file, FORMAT, out_file)
                    bet_command = [os.path.join(*[FSLDIR,'bin', 'bet']), next_file + FORMAT, out_file, "-m", "-S"]
                    
                    _ = subprocess.call(bet_command)
                    
                next_file = out_file
                print("outfile post brain extract: ", out_file)
            
            # ======================================================================================
            # NORMALIZE 
            # ======================================================================================    
            if not islabel:
                # do the normalizing
                img, header = load_nii_img(next_file)
                img = img.squeeze()
                normalize_brain(img) # in place operation

                # save the results
                out_file = os.path.join(output_dir, f"{output_filename}_NORMALIZE")
                save_nii_img(out_file, img, header)
                
                next_file = out_file
                print("outfile post normalize: ", out_file)
    
            # ======================================================================================
            # RESAMPLE
            # ======================================================================================
            out_file = os.path.join(output_dir, output_filename) # last step in preprocessing order
            resample_and_save(next_file, out_file+ FORMAT, is_label=islabel, out_spacing=outspacing, overwrite=True)
            print("outfile post resample: ", out_file)
        
        # resample the brain mask
        flair_outdir = ind_filemap["FLAIR"]["outpath"]
        flair_outfilename = ind_filemap["FLAIR"]["outfilename"]
        mask_file = os.path.join(flair_outdir, flair_outfilename + "_BET_mask" + FORMAT)
        resample_and_save(mask_file, mask_file, is_label=True, out_spacing=outspacing, overwrite=True)
        print()
        
if __name__ == '__main__':
    parser = construct_parser()
    args = parser.parse_args()
    main(args)
            
    
