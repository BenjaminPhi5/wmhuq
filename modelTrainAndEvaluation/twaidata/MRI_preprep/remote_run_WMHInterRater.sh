source ~/.bashrc
source activate wmh

# note this is for the InterRater component of the WMH challenge dataset only.

in_dir=/home/s2208943/datasets/WMH_Challenge
out_dir=/home/s2208943/preprocessed_data/WMHChallenge_InterRaterData

# process WMH Challenge dataset
python simple_preprocess_st2.py -i ${in_dir} -o ${out_dir} -n WMH_InterRater -s 0 -e -1 -f False -z True -a False -k True -l 0.05 -u 0.95