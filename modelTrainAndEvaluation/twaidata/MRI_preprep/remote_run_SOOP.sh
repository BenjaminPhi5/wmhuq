source ~/.bashrc
source activate wmh

in_dir=/home/s2208943/datasets/SOOP
out_dir=/home/s2208943/preprocessed_data/SOOP

# process SOOP dataset
python simple_preprocess_st2.py -i ${in_dir} -o ${out_dir} -n SOOP -s 0 -e -1 -f False -z True -a False -k True -l 0.05 -u 0.95