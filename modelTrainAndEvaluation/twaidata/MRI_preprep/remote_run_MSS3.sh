source ~/.bashrc
source activate wmh

in_dir=/home/s2208943/datasets/Inter_observer
out_dir=/home/s2208943/preprocessed_data/MSS3_InterRaterData

# process MSS3 dataset
python simple_preprocess_st2.py -i ${in_dir} -o ${out_dir} -n MSS3 -s 0 -e -1 -f False -z True -a False -k False -l 0.05 -u 0.95