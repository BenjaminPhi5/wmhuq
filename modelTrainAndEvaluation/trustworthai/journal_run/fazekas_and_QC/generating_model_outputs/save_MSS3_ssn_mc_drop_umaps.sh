#!/bin/bash

source ~/.bashrc
source activate wmh


### ssn
python ../../evaluation/new_scripts/run_model_evaluations.py --repo_dir=/home/s2208943/ipdis/WMH_UQ_assessment --result_dir=/home/s2208943/preprocessed_data/MSS3_InterRaterData/output_maps --ckpt_dir=/home/s2208943/ipdis/results/journal_models/cross_validated_models --script_loc=trustworthai/journal_run/new_MIA_fazekas_and_QC/generating_model_outputs/saving_MSS3_umaps_script.py --uncertainty_type=ssn --model_name=ssn_ens0_cv0

### mc_drop
python ../../evaluation/new_scripts/run_model_evaluations.py --repo_dir=/home/s2208943/ipdis/WMH_UQ_assessment --result_dir=/home/s2208943/preprocessed_data/MSS3_InterRaterData/output_maps --ckpt_dir=/home/s2208943/ipdis/results/journal_models/cross_validated_models --script_loc=trustworthai/journal_run/new_MIA_fazekas_and_QC/generating_model_outputs/saving_MSS3_umaps_script.py --uncertainty_type=mc_drop --model_name=mcdrop_all_0.1_cv0


