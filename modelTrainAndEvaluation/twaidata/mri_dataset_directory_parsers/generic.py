"""
generic Directory Parser, scours a directory given a path to that directory,
looking for files according to the dataset structure, mapping each filepath to
a domain, img or label, and file type
"""
import os
from pathlib import Path
from abc import ABC, abstractmethod

class DirectoryParser(ABC):
    def __init__(self, dataset_root_in, dataset_root_out):
        """
        dataset_root_in: the root directory of the dataset on the filesystem
        dataset_root_out: the location the files should be stored in after they are preprocessed
        """
        if not os.path.exists(dataset_root_in):
            raise FileNotFoundError(f"directory of dataset does not exist: {dataset_root_in}")
            
        if not os.path.exists(dataset_root_out):
            try:
                # os.mkdir(dataset_root_out, exist_ok=True)
                Path(dataset_root_out).mkdir(parents=True, exist_ok=True)
            except FileNotFoundError:
                print(f"Warning: couldn't make output directory here: {dataset_root_out}")
                
        self.root_in = dataset_root_in
        self.root_out = dataset_root_out
        self.files_map = {} # maps each individual to the following file information for each file:{'outdirectory', 'outfilename', 'islabel', '<optional domain key>'}
        self.domain_map = {} # maps each individual to a particular domain. This is optional to use.
        
        self._build_dataset_table()
        
    @abstractmethod
    def _build_dataset_table(self):
        """
        verifies the dataset structure is as expected and builds a map of
        raw files to output
        """
        pass
    
    def get_dataset_inout_map(self):
        """
        returns the mapping from input files to their corresponding output locations, and whether they are
        a label or not (labels are preprocessed differently)
        """
        return self.files_map
    
    def get_domain_map(self):
        return self.domain_map
        