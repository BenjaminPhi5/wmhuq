To perform the intial brain extraction experiment, I used the BET tool from FSL. The guide to download and install it is: https://fsl.fmrib.ox.ac.uk/fsl/fslwiki

the user guide for BET is: https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/BET/UserGuide

the environment variable `${FSLDIR}` says where it is installed, after restart.

I ran `./${FSLDIR}/bin/bet2 <in file> <out file>`

see the visualize files after brain extraction notebook, it seems to match the FLAIR_brain files found in the wmh ../orig folders. Can now apply it over all my data, to T1 and to other datasets. Nice.