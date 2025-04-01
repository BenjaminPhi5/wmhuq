source ~/.bashrc
source activate wmh

in_dir=/home/s2208943/datasets/ISLES2022/ISLES-2022
out_dir=/home/s2208943/preprocessed_data/ISLES2022/
csv_path=/home/s2208943/ipdis/WMH_UQ_assessment/twaidata/ISLES2022_initial_preprocessing/ISLES2022_preprocessing_io_table.csv

# process MSS3 dataset
python simple_preprocess_st2.py -i ${in_dir} -o ${out_dir} -n ISLES2022_from_csv -s 0 -e -1 -f False -z True -a False -k False -l 0.05 -u 0.95 -c ${csv_path} -b False
