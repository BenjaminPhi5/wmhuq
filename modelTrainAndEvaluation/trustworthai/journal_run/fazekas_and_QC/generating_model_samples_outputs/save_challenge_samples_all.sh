#!/bin/bash

source ~/.bashrc
source activate wmh

###training
### punet
python ../../evaluation/new_scripts/run_model_evaluations.py --repo_dir=/home/s2208943/ipdis/WMH_UQ_assessment --result_dir=/home/s2208943/preprocessed_data/WMHChallenge_InterRaterData/output_maps/training --ckpt_dir=/home/s2208943/ipdis/results/journal_models/cross_validated_models --script_loc=trustworthai/journal_run/new_MIA_fazekas_and_QC/generating_model_samples_outputs/saving_WMHCHAL_train_samples_script.py --uncertainty_type=punet --model_name=punet_cv0

### ssn ens
python ../../evaluation/new_scripts/run_model_evaluations.py --repo_dir=/home/s2208943/ipdis/WMH_UQ_assessment --result_dir=/home/s2208943/preprocessed_data/WMHChallenge_InterRaterData/output_maps/training --ckpt_dir=/home/s2208943/ipdis/results/journal_models/cross_validated_models --script_loc=trustworthai/journal_run/new_MIA_fazekas_and_QC/generating_model_samples_outputs/saving_WMHCHAL_train_samples_script.py --uncertainty_type=ssn_ens --model_name=ssn_ens0_cv0

###test
### punet
python ../../evaluation/new_scripts/run_model_evaluations.py --repo_dir=/home/s2208943/ipdis/WMH_UQ_assessment --result_dir=/home/s2208943/preprocessed_data/WMHChallenge_InterRaterData/output_maps/test --ckpt_dir=/home/s2208943/ipdis/results/journal_models/cross_validated_models --script_loc=trustworthai/journal_run/new_MIA_fazekas_and_QC/generating_model_samples_outputs/saving_WMHCHAL_test_samples_script.py --uncertainty_type=punet --model_name=punet_cv0

### ssn ens
python ../../evaluation/new_scripts/run_model_evaluations.py --repo_dir=/home/s2208943/ipdis/WMH_UQ_assessment --result_dir=/home/s2208943/preprocessed_data/WMHChallenge_InterRaterData/output_maps/test --ckpt_dir=/home/s2208943/ipdis/results/journal_models/cross_validated_models --script_loc=trustworthai/journal_run/new_MIA_fazekas_and_QC/generating_model_samples_outputs/saving_WMHCHAL_test_samples_script.py --uncertainty_type=ssn_ens --model_name=ssn_ens0_cv0