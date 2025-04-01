source ~/.bashrc
source activate wmh

in_dir=/home/s2208943/preprocessed_data/

python ventricle_extraction_v2.py -i ${in_dir} -f False