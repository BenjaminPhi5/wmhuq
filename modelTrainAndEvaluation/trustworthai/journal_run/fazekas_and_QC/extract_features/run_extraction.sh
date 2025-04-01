#!/bin/bash
source ~/.bashrc
source activate wmh

python /home/s2208943/ipdis/WMH_UQ_assessment/trustworthai/journal_run/new_MIA_fazekas_and_QC/extract_features/run_feature_extraction.py --model_name=ssn_ens --dataset_name=ADNI300
python /home/s2208943/ipdis/WMH_UQ_assessment/trustworthai/journal_run/new_MIA_fazekas_and_QC/extract_features/run_feature_extraction.py --model_name=deterministic --dataset_name=ADNI300
python /home/s2208943/ipdis/WMH_UQ_assessment/trustworthai/journal_run/new_MIA_fazekas_and_QC/extract_features/run_feature_extraction.py --model_name=punet --dataset_name=ADNI300

python /home/s2208943/ipdis/WMH_UQ_assessment/trustworthai/journal_run/new_MIA_fazekas_and_QC/extract_features/run_feature_extraction.py --model_name=ssn_ens --dataset_name=Challenge
python /home/s2208943/ipdis/WMH_UQ_assessment/trustworthai/journal_run/new_MIA_fazekas_and_QC/extract_features/run_feature_extraction.py --model_name=deterministic --dataset_name=Challenge
python /home/s2208943/ipdis/WMH_UQ_assessment/trustworthai/journal_run/new_MIA_fazekas_and_QC/extract_features/run_feature_extraction.py --model_name=punet --dataset_name=Challenge

python /home/s2208943/ipdis/WMH_UQ_assessment/trustworthai/journal_run/new_MIA_fazekas_and_QC/extract_features/run_feature_extraction.py --model_name=ssn_ens --dataset_name=Ed_CVD
python /home/s2208943/ipdis/WMH_UQ_assessment/trustworthai/journal_run/new_MIA_fazekas_and_QC/extract_features/run_feature_extraction.py --model_name=deterministic --dataset_name=Ed_CVD
python /home/s2208943/ipdis/WMH_UQ_assessment/trustworthai/journal_run/new_MIA_fazekas_and_QC/extract_features/run_feature_extraction.py --model_name=punet --dataset_name=Ed_CVD

python /home/s2208943/ipdis/WMH_UQ_assessment/trustworthai/journal_run/new_MIA_fazekas_and_QC/extract_features/run_feature_extraction.py --model_name=ssn_ens --dataset_name=MSS3
python /home/s2208943/ipdis/WMH_UQ_assessment/trustworthai/journal_run/new_MIA_fazekas_and_QC/extract_features/run_feature_extraction.py --model_name=deterministic --dataset_name=MSS3
python /home/s2208943/ipdis/WMH_UQ_assessment/trustworthai/journal_run/new_MIA_fazekas_and_QC/extract_features/run_feature_extraction.py --model_name=punet --dataset_name=MSS3
