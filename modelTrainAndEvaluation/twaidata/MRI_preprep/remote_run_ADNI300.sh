source ~/.bashrc
source activate wmh

in_dir=/home/s2208943/datasets/ADNI300/ADNI300_selected_data
out_dir=/home/s2208943/preprocessed_data/ADNI300
csv_path=/home/s2208943/ipdis/WMH_UQ_assessment/twaidata/ADNI300_preprocessing/ADNI_300_preprocessing_io_table.csv

# process MSS3 dataset
python simple_preprocess_st2.py -i ${in_dir} -o ${out_dir} -n ADNI300_from_csv -s 0 -e -1 -f False -z True -a False -k True -l 0.05 -u 0.95 -c ${csv_path}
