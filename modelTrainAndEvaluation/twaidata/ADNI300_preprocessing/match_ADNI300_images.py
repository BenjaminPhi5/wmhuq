"""
script for finding matching FLAIR and T1 pairs for a specific date, which also have a Fazekas categorization.
"""

import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import nibabel as nib
import matplotlib.pyplot as plt
import SimpleITK as sitk
from natsort import natsorted

def match_flair_scans_to_annotation_date_spreadsheet(annotation_date_ds, adni_dir):
    print("match flairs to Fazekas spreadsheet")
    
    ID_flair_matches = {}
    ID_date_matches = {}
    flair_folder = adni_dir + "ADNI_data/"
    fails = 0
    partial_successes = 0

    for idx in range(len(annotation_date_ds)):
        # ID is of form ADNI_<code> where <code> could be 002_S_0729 for example
        ID = annotation_date_ds.iloc[idx][0]
        ID_code = "_".join(ID.split("_")[1:])

        scan_date = str(annotation_date_ds.iloc[idx][1])
        scan_date_split = f"{scan_date[0:4]}-{scan_date[4:6]}-{scan_date[6-8]}"

        # get the folder for the particular individual
        ind_folder = f"{flair_folder}{ID_code}/"
        if not os.path.exists(ind_folder):
            print(f"failed: NO ID match for {ID}, {scan_date}: ind_folder: {ind_folder}")
            fails += 1
            continue

        # within the individual folder, look for any folder containing flair
        sub_ind_folders = os.listdir(ind_folder)
        flair_ind_folders = []
        for f in sub_ind_folders:
            if os.path.isdir(ind_folder + f) and 'flair' in f.lower():
                flair_ind_folders.append(f"{ind_folder}{f}/")

        if len(flair_ind_folders) == 0:
            print(f"failed: no flair folder for {ID}")
            fails += 1
            continue

        # within the flair folder, there are separate folders for each date at which
        # a scan is taken.
        flair_dates = [(date, fif) for fif in flair_ind_folders for date in os.listdir(fif)]
        # print(flair_dates)

        selected_flair_folder = None
        selected_date = None
        for (fdate, fif) in flair_dates:
            if scan_date_split in fdate:
                selected_flair_folder = f"{fif}{fdate}/"
                selected_date = fdate
                break

        # if we don't find an exact match for the date, try matching for the same year and month
        if selected_flair_folder == None:
            print(scan_date_split[0:7])
            # try matching on only year and month
            for (fdate, fif) in flair_dates:
                if scan_date_split[0:7] in fdate:
                    selected_flair_folder = f"{fif}{fdate}/"
                    selected_date = fdate
                    partial_successes += 1
                    break
                    
        # if we don't find a match for year and month, try year only
        # with the inclusion of this rule, we get no fails when trying to find a matching flair.
        # nice. however, I should inform maria that I have done this....
        if selected_flair_folder == None:
            print(scan_date_split[0:4])
            # try matching on only year and month
            for (fdate, fif) in flair_dates:
                if scan_date_split[0:4] in fdate:
                    selected_flair_folder = f"{fif}{fdate}/"
                    selected_date = fdate
                    partial_successes += 1
                    break

        if selected_flair_folder == None:
            print(f"failed: no date match for {ID}, {scan_date}")
            print(f"target: {scan_date_split}, given dates: {[d[0] for d in flair_dates]}")
            fails += 1
            continue
            
        # within each date folder, there is a single folder with a uid as its name.
        # we select this folder and finally, we can search for the flair folder inside.
        # we select the .hdr file, which nibabel will use to automatically load the corresponding
        # .img file.
        uid_flair_folder = selected_flair_folder + os.listdir(selected_flair_folder)[0]
        uid_files = os.listdir(uid_flair_folder)
        flair_file = None
        for f in uid_files:
            if ".hdr" in f:
                flair_file = f"{uid_flair_folder}/{f}"

        if flair_file == None:
            print(f"failed: couldn't find flair in date subfolder: {ID}")
            fails += 1

        #print(f"success: {ID}: {flair_file}")
        print("success")

        #print(ID, scan_date)
    
        ID_flair_matches[ID] = flair_file
        ID_date_matches[ID] = selected_date.replace('-', '')[0:8]
    
    return ID_flair_matches, ID_date_matches, fails, partial_successes

def match_t1_to_flair_date(ID_to_flair_date_map, adni_dir):
    print("match T1s to FLAIRs")
    
    ID_t1_matches = {}
    missing_ids = []
    fails = 0
    
    # t1 folder now the same as the flair folder
    t1_folder = adni_dir + "ADNI_data/"
    T1_ids = os.listdir(t1_folder)
    
    for ID, scan_date in ID_to_flair_date_map.items():
        # ID is of form ADNI_<code> where <code> could be 002_S_0729 for example
        ID_code = "_".join(ID.split("_")[1:])
        scan_date_split = f"{scan_date[0:4]}-{scan_date[4:6]}-{scan_date[6-8]}"
        
        if ID_code not in T1_ids:
            missing_ids.append(ID)
            print(f"failed, missing ID: {ID}")
            fails += 1
            
        # I should now be able to find an exact match for each flair that I have found
        # within the individual folder, look for any folder containing MPRAGE
        ind_folder = f"{t1_folder}{ID_code}/"
        sub_ind_folders = os.listdir(ind_folder)
        mprage_ind_folders = []
        for f in natsorted(sub_ind_folders):
            if os.path.isdir(ind_folder + f) and ('mprage' in f.lower() or 'mp-rage' in f.lower()):
                mprage_ind_folders.append(f"{ind_folder}{f}/")

        if len(mprage_ind_folders) == 0:
            print(f"failed: no t1 folder for {ID}")
            fails += 1
            continue
            
        selected_date_folder = None
        for mprage_folder in mprage_ind_folders:
            date_folders = [f"{mprage_folder}{date}/" for date in os.listdir(mprage_folder)]
            for fdate in date_folders:
                if scan_date_split in fdate:
                    selected_date_folder = fdate
                    break
            if selected_date_folder:
                break
        
        if selected_date_folder == None:
            print(f"failed: no date match for {ID} and flair date {scan_date}")
            fails += 1
            continue
        
        # each date folder contains a single folder with a uid. All files are within
        # that folder
        uid_folder = f"{selected_date_folder}{os.listdir(selected_date_folder)[0]}/"
        uid_files = os.listdir(uid_folder)
        t1_file = None
        for f in uid_files:
            if ".hdr" in f:
                t1_file = f"{uid_folder}/{f}"

        if t1_file == None:
            print(f"failed: couldn't find flair in date subfolder: {ID}")
            fails += 1
            continue
            
        else:
            ID_t1_matches[ID] = t1_file
            print(f"success {ID}")
            
        # break
    print(f"fails: {fails}")
    
    return ID_t1_matches
