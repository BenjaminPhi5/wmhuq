source ~/.bashrc
source activate wmh

in_dir=/home/s2208943/datasets/MSSEG
out_dir=/home/s2208943/preprocessed_data/MSSEG_resample1-.6-.6

# process MSS3 dataset
python simple_preprocess_st2.py -i ${in_dir} -o ${out_dir} -n MSSEG -s 0 -e -1 -f False -z True -a False -k True -m True -l 0.05 -u 0.95 -b False --out_spacing 1.,.6,.6
