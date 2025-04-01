in_dir_wmhchal=/media/benp/NVMEspare/datasets/preprocessing_attempts/local_results
out_dir_wmhchal=/media/benp/NVMEspare/datasets/preprocessing_attempts/local_results/collated

# stage two preprocessing - collate each domain into a single file for all imgs and one for all labels
python preprocess_file_collation.py -i ${in_dir_wmhchal} -o ${out_dir_wmhchal} -n WMH_challenge_dataset -d Singapore -H 224 -W 160 -l 1
python preprocess_file_collation.py -i ${in_dir_wmhchal} -o ${out_dir_wmhchal} -n WMH_challenge_dataset -d Utrecht -H 224 -W 160 -l 1
python preprocess_file_collation.py -i ${in_dir_wmhchal} -o ${out_dir_wmhchal} -n WMH_challenge_dataset -d GE3T -H 224 -W 160 -l 1