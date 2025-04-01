source ~/.bashrc
source activate wmh

# note this is for the InterRater component of the WMH challenge dataset only.

in_dir=/home/s2208943/datasets/WMH_Challenge
out_dir=/home/s2208943/preprocessed_data/WMHChallenge_InterRaterData

# process WMH Challenge dataset
python preprocess_file_collation_v2.py -i ${in_dir} -o ${out_dir} -n WMH_InterRater -H 224 -W 160 -d training_Singapore
python preprocess_file_collation_v2.py -i ${in_dir} -o ${out_dir} -n WMH_InterRater -H 224 -W 160 -d training_Utrecht
python preprocess_file_collation_v2.py -i ${in_dir} -o ${out_dir} -n WMH_InterRater -H 224 -W 160 -d training_Amsterdam_GE3T
python preprocess_file_collation_v2.py -i ${in_dir} -o ${out_dir} -n WMH_InterRater -H 224 -W 160 -d test_Singapore
python preprocess_file_collation_v2.py -i ${in_dir} -o ${out_dir} -n WMH_InterRater -H 224 -W 160 -d test_Utrecht
python preprocess_file_collation_v2.py -i ${in_dir} -o ${out_dir} -n WMH_InterRater -H 224 -W 160 -d test_Amsterdam_GE3T
python preprocess_file_collation_v2.py -i ${in_dir} -o ${out_dir} -n WMH_InterRater -H 224 -W 160 -d test_Amsterdam_GE1T5
python preprocess_file_collation_v2.py -i ${in_dir} -o ${out_dir} -n WMH_InterRater -H 224 -W 160 -d test_Amsterdam_Philips_VU_PETMR_01