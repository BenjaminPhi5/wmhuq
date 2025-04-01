source ~/.bashrc
source activate wmh

in_dir=/home/s2208943/datasets/ADNI300/ADNI300_selected_data
out_dir=/home/s2208943/preprocessed_data/ADNI300
csv_path=/home/s2208943/ipdis/WMH_UQ_assessment/twaidata/ADNI300_preprocessing/ADNI_300_preprocessing_io_table.csv

# process MSS3 dataset
python preprocess_file_collation_v2.py -i ${in_dir} -o ${out_dir} -n ADNI300_from_csv -c ${csv_path} -H 224 -W 160

