#!/bin/bash
source ~/.bashrc
source activate wmh

python ../../evaluation/new_scripts/run_model_evaluations.py --repo_dir=/home/s2208943/ipdis/WMH_UQ_assessment --result_dir=/home/s2208943/preprocessed_data/ADNI300/ADNI_300_output_maps --ckpt_dir=/home/s2208943/ipdis/results/journal_models/cross_validated_models --script_loc=trustworthai/journal_run/new_MIA_fazekas_and_QC/generating_model_samples_outputs/saving_ADNI_samples_script.py --uncertainty_type=ens --model_name=ens0_cv0
