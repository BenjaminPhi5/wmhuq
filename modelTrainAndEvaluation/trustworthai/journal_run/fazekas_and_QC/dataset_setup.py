import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.feature_selection import SelectKBest
from natsort import natsorted
from collections import defaultdict
from tqdm import tqdm

from trustworthai.journal_run.new_MIA_fazekas_and_QC.general_utils import *
from trustworthai.journal_run.new_MIA_fazekas_and_QC.combine_all_spreadsheets_together.combining_spreadsheets_together import *

# loading the spredsheets for each dataset
def load_spreadsheets(include_MSS3=False):
    # step 1 prep challenge
    ds_name = "Challenge"
    clinscores_path = "/home/s2208943/preprocessed_data/WMHChallenge_InterRaterData/clinscore_data.csv"
    sample_div_folder = "/home/s2208943/preprocessed_data/WMHChallenge_InterRaterData/feature_spreadsheets"
    extracted_features_folder = "/home/s2208943/preprocessed_data/WMHChallenge_InterRaterData/feature_spreadsheets"

    ssnens_challenge_df, clinscores_challenge_df = load_and_merge_dfs(ds_name, clinscores_path, sample_div_folder, extracted_features_folder, "ssn_ens")
    punet_challenge_df, _ = load_and_merge_dfs(ds_name, clinscores_path, sample_div_folder, extracted_features_folder, "punet")
    deterministic_challenge_df, _ = load_and_merge_dfs(ds_name, clinscores_path, sample_div_folder, extracted_features_folder, "deterministic")

    challenge_data = {
        "ssn_ens":ssnens_challenge_df,
        "punet":punet_challenge_df,
        "deterministic":deterministic_challenge_df
    }
    
    print("challenge shapes before and after cleanup")
    print(challenge_data['ssn_ens'].shape)
    for df in challenge_data.values():
        cleanup_df(df)
    print(challenge_data['ssn_ens'].shape)

    # step 2 prep cvd
    ds_name = "CVD"
    clinscores_path = "/home/s2208943/preprocessed_data/Ed_CVD/clinscore_data.csv"
    sample_div_folder = "/home/s2208943/preprocessed_data/Ed_CVD/EdData_feature_spreadsheets"
    extracted_features_folder = "/home/s2208943/preprocessed_data/Ed_CVD/EdData_feature_spreadsheets"

    ssnens_cvd_df, clinscores_cvd_df = load_and_merge_dfs(ds_name, clinscores_path, sample_div_folder, extracted_features_folder, "ssn_ens")
    punet_cvd_df, _ = load_and_merge_dfs(ds_name, clinscores_path, sample_div_folder, extracted_features_folder, "punet")
    deterministic_cvd_df, _ = load_and_merge_dfs(ds_name, clinscores_path, sample_div_folder, extracted_features_folder, "deterministic")

    cvd_data = {
        "ssn_ens":ssnens_cvd_df,
        "punet":punet_cvd_df,
        "deterministic":deterministic_cvd_df
    }

    print("cvd shapes before and after cleanup")
    print(cvd_data['ssn_ens'].shape)
    for df in cvd_data.values():
        cleanup_df(df)
    print(cvd_data['ssn_ens'].shape)

    # step 3 prep adni

    ds_name = "ADNI300"
    clinscores_path = "/home/s2208943/preprocessed_data/ADNI300/clinscore_data.csv"
    sample_div_folder = "/home/s2208943/preprocessed_data/ADNI300/ADNI_300_feature_spreadsheets"
    extracted_features_folder = "/home/s2208943/preprocessed_data/ADNI300/ADNI_300_feature_spreadsheets"

    ssnens_adni_df, clinscores_adni_df = load_and_merge_dfs(ds_name, clinscores_path, sample_div_folder, extracted_features_folder, "ssn_ens")
    punet_adni_df, _ = load_and_merge_dfs(ds_name, clinscores_path, sample_div_folder, extracted_features_folder, "punet")
    deterministic_adni_df, _ = load_and_merge_dfs(ds_name, clinscores_path, sample_div_folder, extracted_features_folder, "deterministic")

    adni_data = {
        "ssn_ens":ssnens_adni_df,
        "punet":punet_adni_df,
        "deterministic":deterministic_adni_df
    }

    print("adni shapes before and after cleanup")
    print(adni_data['ssn_ens'].shape)
    for df in adni_data.values():
        cleanup_df(df)
    print(adni_data['ssn_ens'].shape)
    
    if not include_MSS3:
        return cvd_data, adni_data, challenge_data
    
    # step 4 prep MSS3
    ds_name = "MSS3"
    clinscores_path = "/home/s2208943/preprocessed_data/MSS3_InterRaterData/clinscore_data.csv"
    sample_div_folder = "/home/s2208943/preprocessed_data/MSS3_InterRaterData/feature_spreadsheets"
    extracted_features_folder = sample_div_folder
    
    ssnens_mss3_df, clinscores_mss3_df = load_and_merge_dfs(ds_name, clinscores_path, sample_div_folder, extracted_features_folder, "ssn_ens")
    punet_mss3_df, _ = load_and_merge_dfs(ds_name, clinscores_path, sample_div_folder, extracted_features_folder, "punet")
    deterministic_mss3_df, _ = load_and_merge_dfs(ds_name, clinscores_path, sample_div_folder, extracted_features_folder, "deterministic")

    mss3_data = {
        "ssn_ens":ssnens_mss3_df,
        "punet":punet_mss3_df,
        "deterministic":deterministic_mss3_df
    }

    print("mss3 shapes before and after cleanup")
    print(mss3_data['ssn_ens'].shape)
    for df in mss3_data.values():
        cleanup_df(df)
    print(mss3_data['ssn_ens'].shape)
    
    return cvd_data, adni_data, challenge_data, mss3_data

def perform_filtering(cvd_data, adni_data, challenge_data, target_fields_list):

    for data in [cvd_data, adni_data, challenge_data]:

        combined_ids = []

        for uncertainty_type in ['ssn_ens', 'punet', 'deterministic']:
            print(f"size before: {uncertainty_type}: {data[uncertainty_type].shape}")

        # get rid of nan data
        for uncertainty_type in ['ssn_ens', 'punet', 'deterministic']:
            X = data[uncertainty_type].copy()

            # remove nan columns
            nan_count = X.isna().sum(axis=0)
            filtered_cols = X.columns[nan_count < 20]
            filtered_cols = filtered_cols.tolist()
            for key in target_fields_list:
                if key not in filtered_cols:
                    filtered_cols.append(key)
            X = X[filtered_cols]

            # remove nan rows
            nan_count = X.isna().sum(axis=1)
            X = X[nan_count < 10]

            # remove all remaining columns that contain nans
            nan_count = X.isna().sum(axis=0)
            filtered_cols = X.columns[nan_count < 1]
            filtered_cols = filtered_cols.tolist()
            for key in target_fields_list:
                if key not in filtered_cols:
                    filtered_cols.append(key)
            X = X[filtered_cols]

            data[uncertainty_type] = X

            combined_ids.append(X.ID.tolist())


        # find the id intersection
        ids_intersection = set(combined_ids[0]) & set(combined_ids[1]) & set(combined_ids[2])

        for uncertainty_type in ['ssn_ens', 'punet', 'deterministic']:
            data[uncertainty_type] = data[uncertainty_type][data[uncertainty_type]['ID'].isin(ids_intersection)]
            print(f"size after: {uncertainty_type}: {data[uncertainty_type].shape}")

        print("-----\n")
        
    return cvd_data, adni_data, challenge_data
    
    
def get_the_data_splits(cvd_data, adni_data, challenge_data, targets_list, models_match_labels=True):
    
    """
    models_match_labels: if true, then for each model (e.g ssn_ens, punet, deterministic), the labels should be identical"
    """
    
    split_config = {'cvd':{'n_splits':3, 'val_proportion':0.3},'adni':{'n_splits':2, 'val_proportion':0.25}}

    split_data = {'WMH_Deep':defaultdict(lambda : (defaultdict(lambda : {}))),
                  'WMH_PV':defaultdict(lambda : (defaultdict(lambda : {}))),
                  'dice_class':defaultdict(lambda : (defaultdict(lambda : {}))),
                  'avd_class':defaultdict(lambda : (defaultdict(lambda : {}))),
                 }

    for target in targets_list:
        for df_set, df_name in zip([cvd_data, adni_data, challenge_data], ['cvd', 'adni', 'challenge']):
            if df_set is None:
                continue
            for uncertainty_type in df_set.keys():
                df = df_set[uncertainty_type]
                X = df.copy()
                X.reset_index()
                # remove rows where the target is nan
                X = X[~np.isnan(X[target])]
                X = X.reset_index()
                y = X[target].values
                # X = X.drop(columns=[target])

                if df_name != 'challenge':
                    (X_train, y_train), (X_test, y_test), (X_val, y_val) = get_fold(
                        X,
                        y,
                        fold_num=0,
                        n_splits=split_config[df_name]['n_splits'],
                        val_proportion=split_config[df_name]['val_proportion']
                    )

                else:
                    X_test, y_test = X, y

                split_data[target][df_name][uncertainty_type]['test'] = (X_test, y_test)

                if df_name != "challenge":
                    split_data[target][df_name][uncertainty_type]['train'] = (X_train, y_train)
                    split_data[target][df_name][uncertainty_type]['val'] = (X_val, y_val)

                if df_name == "adni":
                    # split_data[target][df_name][uncertainty_type]['val'] = (X_val, y_val)

                    # swap the train and the val splits
                    temp = split_data[target][df_name][uncertainty_type]['train']
                    split_data[target][df_name][uncertainty_type]['train'] = split_data[target][df_name][uncertainty_type]['val']
                    split_data[target][df_name][uncertainty_type]['val'] = temp

    # checking the splits have worked properly
    print("checking that the splits have worked properly\nalso a bunch of asserts are run")
    
    if adni_data is not None:
        print("adni ssn_ens train test val sizes")
        train_shape = split_data[targets_list[0]]['adni']['ssn_ens']['train'][0].shape
        test_shape = split_data[targets_list[0]]['adni']['ssn_ens']['test'][0].shape
        val_shape = split_data[targets_list[0]]['adni']['ssn_ens']['val'][0].shape
        print(train_shape);print(test_shape);print(val_shape)
        print(train_shape[0] + test_shape[0] + val_shape[0])

        train_ids = split_data[targets_list[0]]['adni']['ssn_ens']['train'][0]['ID'].values
        test_ids = split_data[targets_list[0]]['adni']['ssn_ens']['test'][0]['ID'].values
        val_ids = split_data[targets_list[0]]['adni']['ssn_ens']['val'][0]['ID'].values
        for tid in test_ids:
            assert tid not in train_ids
        for tid in val_ids:
            assert tid not in train_ids
            assert tid not in test_ids

        if models_match_labels:
            assert np.all(split_data[targets_list[0]]['adni']['ssn_ens']['test'][0].ID == split_data[targets_list[0]]['adni']['punet']['test'][0].ID)
            assert np.all(split_data[targets_list[0]]['adni']['ssn_ens']['train'][0].ID.values == split_data[targets_list[0]]['adni']['deterministic']['train'][0].ID)
    
    if cvd_data is not None:
        print("cvd ssn_ens train test val sizes")
        train_shape = split_data[targets_list[0]]['cvd']['ssn_ens']['train'][0].shape
        val_shape = split_data[targets_list[0]]['cvd']['ssn_ens']['val'][0].shape
        test_shape = split_data[targets_list[0]]['cvd']['ssn_ens']['test'][0].shape
        print(train_shape);print(test_shape);print(val_shape)
        print(train_shape[0] + test_shape[0] + val_shape[0])

        train_ids = split_data[targets_list[0]]['cvd']['ssn_ens']['train'][0]['ID'].values
        test_ids = split_data[targets_list[0]]['cvd']['ssn_ens']['test'][0]['ID'].values
        for tid in test_ids:
            assert tid not in train_ids

        if models_match_labels:
            assert np.all(split_data[targets_list[0]]['cvd']['ssn_ens']['test'][0].ID == split_data[targets_list[0]]['cvd']['punet']['test'][0].ID)
            assert np.all(split_data[targets_list[0]]['cvd']['ssn_ens']['train'][0].ID.values == split_data[targets_list[0]]['cvd']['deterministic']['train'][0].ID)
    
    if challenge_data is not None:
        print("challenge test set size")
        train_shape = split_data[targets_list[0]]['challenge']['ssn_ens']['test'][0].shape
        print(train_shape)
        print(train_shape[0])

        if models_match_labels:
            assert np.all(split_data[targets_list[0]]['challenge']['ssn_ens']['test'][0].ID == split_data[targets_list[0]]['challenge']['punet']['test'][0].ID)
    
    return split_data
