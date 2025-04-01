source ~/.bashrc
source activate wmh

adni_folder=/home/s2208943/datasets/ADNI300/
out_folder=/home/s2208943/datasets/ADNI300/ADNI300_selected_data/
spreadsheet_out_folder=/home/s2208943/ipdis/WMH_UQ_assessment/twaidata/ADNI300_preprocessing/
data_out_folder=/home/s2208943/preprocessed_data/ADNI300

python initial_preprocessing_ADNI300.py -i ${adni_folder} -o ${out_folder} -s ${spreadsheet_out_folder} -d ${data_out_folder}