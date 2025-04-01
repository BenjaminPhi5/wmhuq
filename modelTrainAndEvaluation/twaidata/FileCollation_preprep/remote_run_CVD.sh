source ~/.bashrc
source activate wmh

in_dir=/home/s2208943/datasets/CVD/mixedCVDrelease
out_dir=/home/s2208943/preprocessed_data/Ed_CVD

# process CVD dataset
python preprocess_file_collation_v2.py -i ${in_dir} -o ${out_dir} -n mixedCVDrelease -H 224 -W 160 -d domainA -e None
python preprocess_file_collation_v2.py -i ${in_dir} -o ${out_dir} -n mixedCVDrelease -H 224 -W 160 -d domainB -e None
python preprocess_file_collation_v2.py -i ${in_dir} -o ${out_dir} -n mixedCVDrelease -H 224 -W 160 -d domainC -e None
python preprocess_file_collation_v2.py -i ${in_dir} -o ${out_dir} -n mixedCVDrelease -H 224 -W 160 -d domainD -e None
