Dataset Directory Parsers
--

each raw dataset has a different structure. Each dataset has a specific directory parser that defines where
to find each file in the dataset, its type, and where it should be placed in the final processed dataset.

The dataset parser maps each FLAIR, T1 and mask file to a corresponding file in the output structure:

```
dataset
    - domain 1
        - imgs
            -<id1>_FLAIR.nii.gz
            -<id1>_T1.nii.gz
            ...
        - labels
            -<id1>_wmh.nii.gz
            -<id1>_stroke.nii.gz (optionally)
            ...
     - domain 2
         ...
      
```