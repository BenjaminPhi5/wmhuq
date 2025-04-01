#!/bin/bash

source ~/.bashrc
source activate wmh

### determinstic
python ../../evaluation/new_scripts/run_model_evaluations.py --repo_dir=/home/s2208943/ipdis/WMH_UQ_assessment --result_dir=/home/s2208943/ipdis/test_outs --ckpt_dir=/home/s2208943/ipdis/results/journal_models/cross_validated_models --script_loc=trustworthai/journal_run/new_MIA_fazekas_and_QC/generate_sample_diversity_and_metric_features/saving_samplediv_and_metrics.py --uncertainty_type=deterministic --model_name=ens0_cv0 --dataset=ADNI300

### punet
python ../../evaluation/new_scripts/run_model_evaluations.py --repo_dir=/home/s2208943/ipdis/WMH_UQ_assessment --result_dir=/home/s2208943/ipdis/test_outs --ckpt_dir=/home/s2208943/ipdis/results/journal_models/cross_validated_models --script_loc=trustworthai/journal_run/new_MIA_fazekas_and_QC/generate_sample_diversity_and_metric_features/saving_samplediv_and_metrics.py --uncertainty_type=punet --model_name=punet_cv0 --dataset=ADNI300

### ssn ens
python ../../evaluation/new_scripts/run_model_evaluations.py --repo_dir=/home/s2208943/ipdis/WMH_UQ_assessment --result_dir=/home/s2208943/ipdis/test_outs --ckpt_dir=/home/s2208943/ipdis/results/journal_models/cross_validated_models --script_loc=trustworthai/journal_run/new_MIA_fazekas_and_QC/generate_sample_diversity_and_metric_features/saving_samplediv_and_metrics.py --uncertainty_type=ssn_ens --model_name=ssn_ens0_cv0 --dataset=ADNI300
