#!/bin/bash
source ~/.bashrc
source activate wmh

python /home/s2208943/ipdis/WMH_UQ_assessment/trustworthai/journal_run/new_MIA_fazekas_and_QC/extract_features/run_per_sample_feature_extraction_v2.py --model_name=deterministic --dataset_name=ADNI300

python /home/s2208943/ipdis/WMH_UQ_assessment/trustworthai/journal_run/new_MIA_fazekas_and_QC/extract_features/run_per_sample_feature_extraction_v2.py --model_name=deterministic --dataset_name=Challenge

python /home/s2208943/ipdis/WMH_UQ_assessment/trustworthai/journal_run/new_MIA_fazekas_and_QC/extract_features/run_per_sample_feature_extraction_v2.py --model_name=deterministic --dataset_name=Ed_CVD
