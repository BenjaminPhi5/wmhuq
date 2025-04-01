### steps to loading the dataset

- [x] determine the spacing of the flair image for each image in the wmh datasets I have
- [x] modify the split code so that any visits with a particular image name must be excluded.
- [x] create a dataset object that does random sampling where a subject has multiple visits.
- [x] create the 'in-ram' versions of all of these datasets.
- [ ] preprocess the BTH-MS dataset
- [ ] preprocess the LBC 10 dataset (and remove specific subjects from the LBC for training corresponding to the 12 that have another annotation)
- [ ] determine the evaulations I need (specifically bland altman of volume over ICV volume for best sample and mean sample, focussing on the low volume samples.
- [x] add the monai augmentations to the dataset
- [x] add option to rename the keys to a specific key (e.g image, label, brainmask).
- [ ] now create a training script for the inidividual model data and the combined model data.
- [ ] create a 'policy problem V1 experiment folder to work on this in'

dataset steps:
- [ ] load the spreadsheet
- [ ] filter for any specific sets that should be included / excluded based on specific imgnames or imgtypes
- [ ] determine visit policy (for the in ram model store sub, visit, data on file I think):
    - [ ] all sub-visit sets are separate elements of the datasets (Default)
    - [ ] sample a specific visit each time
    - [ ] longitudinal may need some other logic that we won't do at this stage, just create a wrapper parent class that copies across all the information needed I think
- [x] cropping, padding, resampling:
    - [x] resampling wrapper (so we will need to load using sitk specifically at first, could use sitk for resampling or could try and use torch, not sure which would be faster....).
    - [x] crop and pad to fixed size wrapper
    - [x] crop to mask + a gap (for affine transformation) wrapper
- [ ] load the dataset into ram wrapper
- [ ] map the imgnames to a name to a global name for use later, e.g wmh or lbc1_wmh etc.
- [ ] convert the dataset to MONAI format

- [ ] define the code for all the different models that I want to train. the multi-head model could get tricky (i.e do I do SSN, where do I branch off the head from etc etc etc).
- [ ] I should define a basic ResUNet based on nnUNet I think maybe? Yes.

### longitudinal data
for longitudinal data, I will need to option to load only specific visits (e.g visit 0 and visit 3) and not load all the images from all visits at once.

leave the design of the longitudinal datasets until we have actually run the preprocessing on the longitudinal data since this will take a slightly more careful construction of the experiment.

torch datasets for loading our brain MRI data.

will use monai style dataset.

the first thing I need to do is create a dataset wrapper that is based on loading each image independently from disk.

there will be datasets that run once modifying state (e.g cropping and optional padding etc).

the initial basic pipeline needs to do the following:

- load the dataset based on the subject file manager and the data spreadsheet.
- take a list of imagetypes that it is to include
- take a list of specific image names that it will include
- keep record of image type and image label.
- have an option to load a clinical data spreadsheet that returns a dictionary of information (e.g age, any other information).

extra dataset wrappers that I need to make:

- select a random visit for each subject (will want to be able to do that potentially after the 'in-ram' dataset
- will need a slurm copying function for preprocessed data if I am going to do loading from disk etc on mlp cluster.

- 