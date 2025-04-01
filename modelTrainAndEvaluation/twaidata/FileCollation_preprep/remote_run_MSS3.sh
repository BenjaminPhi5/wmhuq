source ~/.bashrc
source activate wmh

in_dir=/home/s2208943/datasets/Inter_observer
out_dir=/home/s2208943/preprocessed_data/MSS3_InterRaterData

# process MSS3 dataset
python preprocess_file_collation_v2.py -i ${in_dir} -o ${out_dir} -n MSS3 -H 224 -W 160