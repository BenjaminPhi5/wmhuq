import os
from natsort import natsorted

def create_ADNI_spreadsheet(converted_nifti_folder, data_output_folder, spreadsheet_output_folder):
    # should be in format
    # <input file path>,<output file folder>,<output_id>_<image mode>,is_label
    # example from that challenge dataset:
    # 50,Singapore/50/orig/FLAIR.nii.gz,Singapore/imgs/,FLAIR,False,<optional domain key>

    id_folders = os.listdir(converted_nifti_folder)
    
    imgs_output_folder = os.path.join(data_output_folder, "imgs")

    with open(os.path.join(spreadsheet_output_folder, "ADNI_300_preprocessing_io_table.csv"), 'w') as f:
        for idx, ID in enumerate(natsorted(id_folders)):
            example_files = os.listdir(converted_nifti_folder + ID)
            example_folder = converted_nifti_folder + ID + "/"
            flair_example = example_folder + [f for f in example_files if 'flair' in f][0]
            # uses the registered version of the t1.
            t1_example = example_folder + [f for f in example_files if 't1w_registered' in f][0]

            f.write(f"{idx},{ID},{flair_example},{imgs_output_folder},FLAIR,False\n")
            f.write(f"{idx},{ID},{t1_example},{imgs_output_folder},T1,False\n")
