#!/bin/bash

# Specify the directory to process
directory="/media/benp/NVMEspare/model_data/journal_models"

# Specify the Python script to call
python_script="deterministic_model_eval.py"

# Loop over all folder names in the directory
for folder in "$directory"/*; do
    # Check if the path is a directory
    if [ -d "$folder" ]; then
        # Get the folder name
        folder_name=$(basename "$folder")

        # Call the Python script with the folder name as a parameter
        echo "$folder_name"
        python "$python_script" --ckpt_dir=/media/benp/NVMEspare/model_data/journal_models/ --repo_dir=/home/benp/Documents/PhD_projects/WMH_UQ_assessment --result_dir=trustworthai/journal_run/evaluation/results/initial_hparam_tuning/out_domain_results --dataset=chal --eval_split="train" --overwrite=false --model_name="$folder_name"
    fi
done