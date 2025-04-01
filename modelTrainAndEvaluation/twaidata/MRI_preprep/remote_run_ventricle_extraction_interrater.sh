source ~/.bashrc
source activate wmh

# run on MSS3
python ventricle_extraction.py -i /home/s2208943/ipdis/data/preprocessed_data/MSS3_InterRaterData/imgs

# run on LBC
python ventricle_extraction.py -i /home/s2208943/ipdis/data/preprocessed_data/LBC_InterRaterData/imgs

# run on WMH Challenge inter rater version
python ventricle_extraction.py -i /home/s2208943/ipdis/data/preprocessed_data/WMHChallenge_InterRaterData/imgs

# todo: update this so that I store the full synth seg output for all images on all datasets, as well as the ventricle extraction. The synthseg output could be useful later and serves as a useful pretraining task actually.