print("strawberry")
# trainer
from trustworthai.utils.fitting_and_inference.fitters.basic_lightning_fitter import StandardLitModelWrapper
from trustworthai.utils.fitting_and_inference.get_trainer import get_trainer

# data
from twaidata.torchdatasets_v2.mri_dataset_inram import MRISegmentation3DDataset
from twaidata.torchdatasets_v2.mri_dataset_from_file import MRISegmentationDatasetFromFile, ArrayMRISegmentationDatasetFromFile
from twaidata.mri_dataset_directory_parsers.MSS3_multirater import MSS3MultiRaterDataParser
from torch.utils.data import ConcatDataset

# packages
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torchinfo import summary
from tqdm import tqdm
from collections import defaultdict
from natsort import natsorted
import torchmetrics
import argparse

from trustworthai.journal_run.new_MIA_fazekas_and_QC.extract_features.utils import *
print("banana")

def construct_parser():
    parser = argparse.ArgumentParser(description = "extract features for fazekas prediction model")
    
    parser.add_argument('--model_name', required=True)
    parser.add_argument('--dataset_name', required=True)
    
    return parser

def main(args):
    model_name = args.model_name
    
    ds_name = args.dataset_name
    
    if ds_name == "Ed_CVD":
        domains_ed = ["domainA", "domainB", "domainC", "domainD"]
        ds = ConcatDataset([
            MRISegmentation3DDataset("/home/s2208943/preprocessed_data/Ed_CVD/collated", no_labels=True, xy_only=False, domain_name=dn)
            for dn in domains_ed
        ])
        
        output_maps_dirs = [f"/home/s2208943/preprocessed_data/Ed_CVD/EdData_output_maps/{model_name}/"]
        out_folder_name = "/home/s2208943/preprocessed_data/Ed_CVD/EdData_feature_spreadsheets"
        synthseg_dirs = [f'/home/s2208943/preprocessed_data/Ed_CVD/{dn}/imgs/' for dn in domains_ed]
   
    elif ds_name == "ADNI300":
        ds = MRISegmentation3DDataset("/home/s2208943/preprocessed_data/ADNI300/collated", no_labels=True, xy_only=False)
        output_maps_dirs = [f"/home/s2208943/preprocessed_data/ADNI300/ADNI_300_output_maps/{model_name}/"]
        out_folder_name = '/home/s2208943/preprocessed_data/ADNI300/ADNI_300_feature_spreadsheets'
        synthseg_dirs = ['/home/s2208943/preprocessed_data/ADNI300/imgs/']
        
    elif ds_name == "Challenge":
        domains_chal = ["training_Singapore", "training_Utrecht", "training_Amsterdam_GE3T", "test_Amsterdam_GE1T5", "test_Amsterdam_Philips_VU_PETMR_01", "test_Utrecht", "test_Amsterdam_GE3T", "test_Singapore"]
    
        ds = ConcatDataset([
            MRISegmentation3DDataset("/home/s2208943/preprocessed_data/WMHChallenge_InterRaterData/collated", no_labels=True, xy_only=False, domain_name=dn)
            for dn in domains_chal
        ])
        
        output_maps_dirs = [f"/home/s2208943/preprocessed_data/WMHChallenge_InterRaterData/output_maps/training/{model_name}/", f"/home/s2208943/preprocessed_data/WMHChallenge_InterRaterData/output_maps/test/{model_name}/"] 
        out_folder_name = "/home/s2208943/preprocessed_data/WMHChallenge_InterRaterData/feature_spreadsheets"
        synthseg_dirs = [f'/home/s2208943/preprocessed_data/WMHChallenge_InterRaterData/{dn.split("_")[0]}/{"_".join(dn.split("_")[1:])}/imgs/' for dn in domains_chal]
        print(synthseg_dirs)
    elif ds_name == "MSS3":
        parser = MSS3MultiRaterDataParser(
            # paths on the cluster for the in house data
            "/home/s2208943/datasets/Inter_observer",
            "/home/s2208943/preprocessed_data/MSS3_InterRaterData"
        )

        ds = ArrayMRISegmentationDatasetFromFile(parser) 
        output_maps_dirs = [f"/home/s2208943/preprocessed_data/MSS3_InterRaterData/output_maps/{model_name}"]
        out_folder_name = "/home/s2208943/preprocessed_data/MSS3_InterRaterData/feature_spreadsheets/"
        synthseg_dirs = ['/home/s2208943/preprocessed_data/MSS3_InterRaterData/imgs/']
    
    else:
        raise ValueError("ds unknown")

    ### load base dataset
    
    
    IDs = [v[2] for v in ds]

    ### load model outputs
    print("LOADING MODEL OUTPUTS")
    output_maps = {}
    key_order = [] 
    for omd in output_maps_dirs:
        om, k = load_output_maps(omd)
        output_maps.update(om)
        key_order.extend(k)
        
    ### load synthseg outputs
    print("LOADING SYNTHSEG OUTPUTS")
    # added a whole load of extra logic for the challenge dataset. In future, I need to rework the preprocessing so that the Id's match how the rest of the datsets work
    synthseg_outs = []
    vent_dists = []
    failure_ids = []
    for data in tqdm(ds):
        synthseg = None
        vent_d = None
        flair = data[0][0]
        patient_id = data[2]
        # print(patient_id)
        es = []
        if ds_name == "Challenge":
            patient_id = patient_id.split("_")[-1]
            # print(patient_id)
        for folder in synthseg_dirs:
            if ds_name == "Challenge":
                if "_".join(patient_id.split("_")[1:-1]) not in folder:
                    continue
            try:
                synthseg, vent_d = load_synthseg_data(folder, patient_id, flair)
                # print("success")
                break
            except Exception as e:
                es.append(e)
                continue
        if synthseg is None:
            print("failed to load synthseg for : ", patient_id)
            # for e in es:
            #     print(e)
        synthseg_outs.append(synthseg)
        vent_dists.append(vent_d)
    
    ### run analysis script
    print("RUNNING ANALYSIS")
    data = {}
    failed_ids = []
    for idx in tqdm(range(len(IDs)), position=0, leave=True):
        ID = IDs[idx]
        # print(ID)
        try:
            synthseg_map = torch.from_numpy(synthseg_outs[idx]).cuda()
            vmap = torch.from_numpy(vent_dists[idx]).cuda()
            smap = output_maps[IDs[idx]][2].cuda()
            smap_expanded = get_edge_expanded_seg(smap)
            seg_volume = smap.sum().item()

            vent_volume = (synthseg_map == 4).sum().item() + (synthseg_map == 43).sum().item() # 4 and 43 are the keys for the left and right ventricles respectively from the synthseg map.

            ind_data = {}

            for input_type in ["ent", "pred"]:
                if input_type == "ent":
                    umap = output_maps[IDs[idx]][0].cuda()
                    ts = np.arange(0.2, 0.8, 1/20) * 0.7
                    maxv = 0.7
                # elif input_type == "var":
                #     umap = output_maps[IDs[idx][1]][3].cuda()
                #     ts = [0.01, 0.03, 0.06, 0.09, 0.12, 0.15]
                #     maxv = 0.2
                elif input_type == "pred":
                    umap = output_maps[IDs[idx]][1].cuda()
                    # ts = [0.5, 0.25, 0.45, 0.5, 0.65, 0.85]
                    ts = np.arange(0.2, 0.8, 1/20) * 1
                    maxv = 1.0
                else:
                    raise ValueError  

                collected_data_t = {}


                for t in ts:
                    all_data = {}
                    for region in ['all', 'deep', 'pv']:
                        ut = umap > t

                        if region == 'deep':
                            ut = ut * (vmap > 10)
                        elif region == 'pv':
                            ut = ut * (vmap <= 10)
                        else:
                            pass # ut = ut. Nice.

                        umap_values = umap[ut]

                        #### summary statistics
                        sum, mean, std, skew, kurtosis = gaussian_summary_statistics(umap_values.cpu())

                        #### overlap statistics
                        intersection = (ut * smap).sum()
                        expanded_intersection = (ut * smap_expanded).sum()

                        prop_umap_segmented = (intersection / ut.sum()).item()
                        prop_umap_within_expanded_seg = (expanded_intersection / ut.sum()).item()
                        prop_seg_uncertain = (intersection / smap.sum()).item()


                        ### connected component analysis statistics
                        conn_comps = get_conn_comps(ut)
                        cc_data = conn_comp_basic_stats(umap, conn_comps, smap, vmap)
                        cc_data["log_sizes"] = torch.Tensor(cc_data["sizes"]).log()

                        # recorded values
                        region_data = {}
                        for key, value in cc_data.items():
                            region_data["cc_" + key] = value
                        region_data["sum"] = sum
                        region_data["mean"] = mean
                        region_data["std"] = std
                        region_data["skew"] = skew
                        region_data["kurtosis"] = kurtosis
                        region_data["prop_umap_segmented"] = prop_umap_segmented
                        region_data["prop_umap_within_expanded_seg"] = prop_umap_within_expanded_seg
                        region_data["prop_seg_uncertain"] = prop_seg_uncertain

                        region_data = {f"{region}_{key}":value for (key, value) in region_data.items()}
                        for key in region_data:
                            all_data[key] = region_data[key]

                    all_data["seg_volume"] = seg_volume
                    all_data["vent_volume"] = vent_volume
                    
                    for key in all_data.keys():
                        # print(all_data[key])
                        if np.sum(np.array(all_data[key])) == 0:
                            pass
                            # print(f"ZERO ISSUE WITH KEY: {key}_{input_type}_{t}")
                        elif np.all(np.isnan(np.array(all_data[key]))):
                            print(f"NAN VALUE ISSUE WITH KEY: {key}_{input_type}_{t}")

                    # print(all_data["all_cc_num"], all_data["all_cc_unsegmented_ccs"])
                    collected_data_t[f"{t:.2f}"] = all_data


                    # break

                ind_data[input_type] = collected_data_t
                # break

            data[ID] = ind_data
            # break
        except Exception as e:
            print("FAILED FOR : ", ID)
            print(e)
            # raise e
            failed_ids.append((idx, ID))
        # break

    ### save the data
    pd_dfs = defaultdict(lambda : defaultdict(lambda : defaultdict(lambda : [])))
    pd_other_data_dfs = defaultdict(lambda : [])

    for ID_key in tqdm(natsorted(data.keys()), position=0, leave=True):
        ind_data = data[ID_key]
        for input_key in natsorted(ind_data.keys()):
            map_type_data = ind_data[input_key]
            for t_key in natsorted(map_type_data.keys()):
                row_data = map_type_data[t_key]

                #row = []
                pd_dfs[input_key][t_key]['ID'].append(ID_key)
                for key in [
                    'all_cc_num', 'all_cc_unsegmented_ccs', 'all_cc_size_mean', 'all_cc_size_std', 'all_cc_size_skew',
                    'all_cc_size_kurt', 'all_cc_vd_mean', 'all_cc_vd_std', 'all_cc_vd_skew', 'all_cc_vd_kurt', 'all_sum',
                    'all_mean', 'all_std', 'all_skew', 'all_kurtosis', 'all_prop_umap_segmented',
                    'all_prop_umap_within_expanded_seg', 'all_prop_seg_uncertain',
                    'deep_cc_num', 'deep_cc_unsegmented_ccs',
                    'deep_cc_size_mean', 'deep_cc_size_std', 'deep_cc_size_skew', 'deep_cc_size_kurt', 'deep_cc_vd_mean', 'deep_cc_vd_std', 'deep_cc_vd_skew', 'deep_cc_vd_kurt', 'deep_sum',
                    'deep_mean', 'deep_std', 'deep_skew', 'deep_kurtosis', 'deep_prop_umap_segmented', 'deep_prop_umap_within_expanded_seg', 'deep_prop_seg_uncertain',
                    'pv_cc_num',  'pv_cc_unsegmented_ccs', 'pv_cc_size_mean', 'pv_cc_size_std', 'pv_cc_size_skew', 'pv_cc_size_kurt', 'pv_cc_vd_mean', 'pv_cc_vd_std',
                    'pv_cc_vd_skew', 'pv_cc_vd_kurt', 'pv_sum', 'pv_mean', 'pv_std', 'pv_skew', 'pv_kurtosis', 'pv_prop_umap_segmented', 'pv_prop_umap_within_expanded_seg', 'pv_prop_seg_uncertain', 'seg_volume', 'vent_volume']:
                    if 'joint' not in key and 'hist' not in key:
                        #row.append((key, row_data[key]))
                        pd_dfs[input_key][t_key][key].append(row_data[key])
                    else:
                        table_vals = row_data[key].view(-1).tolist()
                        for i in range(len(table_vals)):
                            #row.append((f"{key}_{i}", table_vals[i]))
                            pd_dfs[input_key][t_key][f"{key}_{i}"].append(table_vals[i])

                for key in [
                                'pv_cc_sizes', 'pv_cc_means', 'pv_cc_stds', 'pv_cc_skews', 'pv_cc_kurts', 'pv_cc_vent_dists', 'pv_cc_log_sizes',
                                'deep_cc_sizes', 'deep_cc_means', 'deep_cc_stds', 'deep_cc_skews', 'deep_cc_kurts', 'deep_cc_vent_dists', 'deep_cc_log_sizes',
                                'all_cc_sizes', 'all_cc_means', 'all_cc_stds', 'all_cc_skews', 'all_cc_kurts', 'all_cc_vent_dists', 'all_cc_log_sizes'
                            ]:
                    pd_other_data_dfs[key].extend(row_data[key])
                pd_other_data_dfs["id"].extend([f"{ID_key}_{input_key}_{t_key}" for _ in range(len(row_data['all_cc_sizes']))])
                
    # save the summary statistics
    for input_key in tqdm(natsorted(ind_data.keys()), position=0, leave=True):
        map_type_data = ind_data[input_key]
        for t_key in natsorted(map_type_data.keys()):
            df = pd.DataFrame(pd_dfs[input_key][t_key])
            folder_name = out_folder_name
            if not os.path.exists(folder_name):
                os.makedirs(folder_name)
            df.to_csv(os.path.join(folder_name, f"{model_name}_{input_key}_{t_key}.csv"))
            
    # save the per individual connected component statistics
    np_df = {}
    for key in pd_other_data_dfs.keys():
        np_df[key] = np.array(pd_other_data_dfs[key])
    np.save(os.path.join(folder_name,f"{model_name}_connected_component_data.npy"), np_df)
    
if __name__ == '__main__':
    parser = construct_parser()
    args = parser.parse_args()
    main(args)
