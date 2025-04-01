# script to call the preprocessing code
# in the pgr cluster for the wmh challenge and the ed inhouse datasets

# argument 1 is which individual to start at, argument 2 is which individual to stop at

# paths to datasets on machine
out_dir=/home/s2208943/ipdis/data/preprocessed_data/ADNI_300/
in_dir=/home/s2208943/ipdis/data/ADNI_data/selected_nifti_300/
csv_path=/home/s2208943/ipdis/UQ_WMH_methods/twaidata/ADNI_preprocessing/ADNI_300_preprocessing_io_table.csv

# process WMH dataset
python simple_preprocess_st1.py -i ${in_dir} -o ${out_dir} -n ADNI_300 -c ${csv_path} -s $1 -e $2 -f $3 -z $4

# process ed inhouse dataset
# but ed data is on clusyer only, but this is what i would run.
# python simple_preprocess.py -i ${in_dir_ed} -o ${out_dir} -n WMH_challenge_dataset -s "0" -e "-1"