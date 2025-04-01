"""
match FLAIRs, T1s, Fazekas.

Convert files to nifti

register the T1 to the FLAIR

create the input output spreadsheet for the parser used in the main preprocessing stage

"""

import pandas as pd
import numpy as np
import os
from twaidata.ADNI300_preprocessing.build_fileparser_spreadsheet import create_ADNI_spreadsheet
from twaidata.ADNI300_preprocessing.match_ADNI300_images import match_flair_scans_to_annotation_date_spreadsheet, match_t1_to_flair_date
from twaidata.ADNI300_preprocessing.register_t1s_to_flair import register_all_t1_to_flair
from twaidata.ADNI300_preprocessing.convert_selected_to_nifti import convert_adni_folders_to_nifti
import argparse

def construct_parser():
    # preprocessing settings
    parser = argparse.ArgumentParser(description = "initial preprocessing for ADNI300 subset")
    parser.add_argument('-i', '--adni_folder', required=True, type=str)
    parser.add_argument('-o', '--out_folder', required=True, type=str)
    parser.add_argument('-s', '--spreadsheet_out_folder', required=True, type=str)
    parser.add_argument('-d', '--data_out_folder', required=True, type=str)
    return parser

def main(args):
    adni_dir = args.adni_folder
    spreadsheet_dir = adni_dir
    converted_nifti_folder = args.out_folder
    spreadsheet_out_folder = args.spreadsheet_out_folder
    data_out_folder = args.data_out_folder
    
    # match FLAIRs to fazekas
    print("1/5 matching FLAIRS to fazekas")
    fazekas_csv = pd.read_csv(spreadsheet_dir + "ADNI_data_300sample_Fazekas.csv", header=None)
    ID_flair_matches, ID_date_matches, fails, partial_successes = match_flair_scans_to_annotation_date_spreadsheet(fazekas_csv, adni_dir)

    # match T1s to FLAIRs
    print("\n2/5 matching T1s to FLAIRs")
    ID_t1_matches = match_t1_to_flair_date(ID_date_matches, adni_dir)
    
    # convert files to nifti
    print("\n3/5 Converting files to nifti format and saving to new location")
    convert_adni_folders_to_nifti(ID_flair_matches, ID_t1_matches, ID_date_matches, converted_nifti_folder, force=True, force_individuals=True)
    
    # registering T1 to flair space
    print("\n4/5 Registering all the T1s to flair space")
    ADNI_nifti_ids = os.listdir(converted_nifti_folder)
    register_all_t1_to_flair(converted_nifti_folder, force_individuals=True)
    
    # creating the input output spreadsheet for the main preprocessing step
    print("\n5/5 Creating the input output spreadsheet for the dataset parser")
    create_ADNI_spreadsheet(converted_nifti_folder, data_out_folder, spreadsheet_out_folder)
    
    print("DONE")
    
if __name__ == '__main__':
    parser = construct_parser()
    args = parser.parse_args()
    main(args)

