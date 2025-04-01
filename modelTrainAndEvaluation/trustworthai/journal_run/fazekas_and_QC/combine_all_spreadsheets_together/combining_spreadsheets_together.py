import numpy as np
import pandas as pd
import os

def load_and_merge_dfs(ds_name, clinscores_path, sample_div_folder, extracted_features_folder, model_name):
    # resolve paths
    if ds_name == "CVD":
        sample_div_paths = [os.path.join(extracted_features_folder, f) for f in os.listdir(extracted_features_folder) if "sample_div_and_metrics" in f and model_name in f]
        sample_div_df = pd.read_csv(sample_div_paths[0])
        for sdp in sample_div_paths[1:]:
            sddf = pd.read_csv(sdp)
            sample_div_df = pd.concat([sample_div_df, sddf])
    else:
        sample_div_path = os.path.join(sample_div_folder, f"{model_name}_sample_div_and_metrics.csv")
        sample_div_df = pd.read_csv(sample_div_path)
    extracted_features_files = [os.path.join(extracted_features_folder, f) for f in os.listdir(extracted_features_folder) if ("_pred_" in f or "_ent_" in f) and model_name in f]
    
    # load dfs
    clinscores_df = pd.read_csv(clinscores_path)
    remaining_dfs = [(sample_div_df, "sample_div_and_metrics")] + [(pd.read_csv(path), path) for path in extracted_features_files]
    
    # merge dfs
    merged_df = merge_dfs(clinscores_df, remaining_dfs, ds_name)

    return merged_df, clinscores_df

def merge_dfs(clinscores_df, remaining_dfs, ds_name):
    combined_df = clinscores_df.copy()
    combined_df = combined_df.rename(columns={"Patient ID":"ID"})
    for (df, path) in remaining_dfs:
        df = df.copy()
        try:
            if ds_name == "ADNI300":
                df['ID'] = ["_".join(v.split("_")[1:-1]) for v in df['ID'].values]
            if ds_name == "Challenge":
                df['ID'] = ["_".join(v.split("_")[1:]) for v in df['ID'].values]

            if "_pred_" in path or "_ent_" in path:
                feature_type = "_".join(path[:-4].split("_")[-2:])
                df = df.rename(columns = {key:f"{key}_{feature_type}" for key in df.keys() if key != "ID"})

            combined_df = combined_df.merge(df, on="ID", how='outer')
            if len(combined_df) <= 1:
                print(df.keys())
                print(df['ID'].values)
        except Exception as e:
            print(df.keys())
            print(path)

    return combined_df
