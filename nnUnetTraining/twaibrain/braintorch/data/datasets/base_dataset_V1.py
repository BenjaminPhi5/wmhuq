from twaibrain.braintorch.utils.load_image import torch_load, torch_load_and_resample
from twaibrain.braintorch.utils.fit_to_mask import get_fit_coords, fit_to_mask
from twaibrain.braintorch.utils.resize import crop_or_pad_dims
import os
import numpy as np
import SimpleITK as sitk
from abc import ABC, abstractmethod
from torch.utils.data import Dataset
from twaibrain.braindataconfig.load_config import load_config_data
from tqdm import tqdm

class AbstractBrainDataset_V1(Dataset, ABC):
    def __init__(
        self,
        dataset_folder,
        config_name,
        ds_name,
        visit_key,
        ds_experiment_name='',
        split='train',
        resample=False,
        out_spacing=None,
        fit_to_mask=False,
        mask_pad=0,
        fit_to_shape=False,
        output_shape=[128,128,128],
        key_renames=None,
    ):
        """
        dataset_folder: base path to the preprocessed datasets folder
        
        config_name, ds_name, visit_key, ds_experiment_name: define the data config that will be loaded
        
        visit_key: which column to use to divide the subject images up into visits

        split: one of : train | val | test (which set of subjects to use)

        key_renames: (optional) names of images who's name appear in this dictionary as a key are renamed to the corresponding value, e.g so that wave1-WMH and wave2-WMH from different visits can be renamed just to WMH.
        """

        config_data = load_config_data(config_name, ds_name, ds_experiment_name)
        df = config_data['imgs_df']
        subjects = config_data[f'{split}_subjects']
        self.ds_name = ds_name

        # configure outfilepaths
        df['outfilepath'] = [os.path.join(dataset_folder, ds_name, fp) for fp in df['outfilepath'].values]

        self.df = df
        self.visit_key = visit_key
        self.subjects = sorted([str(s) for s in subjects['subjects'].values])
        self.filter_visit = False
        if 'visit' in subjects.columns:
            self.filter_visit = True
            self.subjects_df = subjects

        # initial preprocessing steps
        self.resample = resample
        self.out_spacing = out_spacing
        self.fit_to_mask = fit_to_mask
        self.mask_pad = mask_pad
        self.fit_to_shape = fit_to_shape
        self.output_shape = output_shape

        if self.resample and self.out_spacing is None:
            raise ValueError("out_spacing must be defined if resample is True")

        self.key_renames = key_renames

    def __len__(self):
        return self.N

    def _load_into_ram(self, index_key, keys):
        self.loaded_data = {}
        df = self.df
        for key in tqdm(keys, ncols=100):
            imgs_df = df[df[index_key] == key]

            data_dict = self._load_imgsdf(imgs_df)
            self.loaded_data[key] = data_dict

    def _load_imgsdf(self, imgs_df):
        """
        function to load all the images in a dataframe
        """
        data_dict = {
            (imgname if is_label else imgtype): (path, is_label) for (imgname, imgtype, path, is_label) in imgs_df[['imgname', 'imgtype', 'outfilepath', 'is_label']].values
        }

        # load image
        data_dict = {key: torch_load_and_resample(path, self.out_spacing, is_label=is_label) 
                     if self.resample else 
                     torch_load(path)
                     for (key, (path, is_label)) in data_dict.items()
                    }

        # fit to mask
        if self.fit_to_mask:
            mask_img = data_dict['brainmask']
            zs, xs, ys = get_fit_coords(mask_img, self.mask_pad)
            data_dict = {key: fit_to_mask(zs, xs, ys, img) for key, img in data_dict.items()}

        # crop or pad to shape
        if self.fit_to_shape:
            data_dict = {key: crop_or_pad_dims(img, [1, 2, 3], self.output_shape) for key, img in data_dict.items()}

        if self.key_renames is not None:
            data_dict = {
                (key if key not in self.key_renames else self.key_renames[key]) : value for (key, value) in data_dict.items()
            }

        return data_dict

class SingleVisitDataset_V1(AbstractBrainDataset_V1):
    def __init__(self, *args, **kwargs):
        """
        a basic wrapper on the AbstractBrainDataset that assumes that each subject comes with a single visit
        and loads each image in X in alphabetical order of the imgnames
        """
        super().__init__(*args, **kwargs)
        
        self.N = len(self.subjects)
    
    def __getitem__(self, idx):
        sub = self.subjects[idx]
        imgs_df = self.df[self.df['sub'] == sub]

        if self.filter_visit:
            visit = self.subjects_df[self.subjects_df['subjects'] == sub]['visit'].values[0]
            imgs_df =imgs_df[imgs_df[self.visit_key] == visit]

        data_dict = self._load_imgsdf(imgs_df)

        # aux_df = imgs_df[imgs_df['imgname'].isin(self.auxfiles)].sort_values(by='imgname')
        # non_aux_df = imgs_df[~imgs_df['imgname'].isin(self.auxfiles)]
        
        # xs_df = non_aux_df[non_aux_df['imgtype'] != 'label'].sort_values(by='imgtype')
        # ys_df = non_aux_df[non_aux_df['imgtype'] == 'label'].sort_values(by='imgname')

        # xs_data = {imgname, data_dict[imgname] imgname in  xs_df['imgname'].values}
        # ys_data = {imgname, data_dict[imgname] imgname in  ys_df['imgname'].values}
        # aux_data = {imgname, data_dict[imgname] imgname in  aux_df['imgname'].values}

        return data_dict

class SingleVisitDatasetInRam_V1(SingleVisitDataset_V1):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._load_into_ram('sub', self.subjects)
        
    def __getitem__(self, idx):
        sub = self.subjects[idx]
        return self.loaded_data[sub]

class SitkImageDataset_V1(SingleVisitDataset_V1):
    def __getitem__(self, idx):
        sub = self.subjects[idx]
        imgs_df = self.df[self.df['sub'] == sub]

        if self.filter_visit:
            visit = self.subjects_df[self.subjects_df['subjects'] == sub]['visit'].values[0]
            imgs_df =imgs_df[imgs_df[self.visit_key] == visit]
        
        data_dict = {
            (imgname if is_label else imgtype): (path, is_label) for (imgname, imgtype, path, is_label) in imgs_df[['imgname', 'imgtype', 'outfilepath', 'is_label']].values
        }
        
        # load image
        data_dict = {key: sitk.ReadImage(path) for (key, (path, _)) in data_dict.items()}
        
        
        return data_dict

class MultiIndependentVisitDataset_V1(AbstractBrainDataset_V1):
    def __init__(self, *args, **kwargs):
        """
        a basic wrapper on the AbstractBrainDataset that assumes that each subject-visit pair is a separate element of the dataset
        """
        super().__init__(*args, **kwargs)

        self.df['key'] = [f'{sub}_{visit}' for (sub, visit) in self.df[['sub', self.visit_key]].values]
        self.keys = sorted([str(k) for k in self.df['key'].unique()])
        self.N = len(self.keys)
    
    def __getitem__(self, idx):
        key = self.keys[idx]
        imgs_df = self.df[self.df['key'] == key]

        data_dict = self._load_imgsdf(imgs_df)
        
        return data_dict

class MultiIndependentVisitDatasetInRam_V1(MultiIndependentVisitDataset_V1):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._load_into_ram('key', self.keys)
        
    def __getitem__(self, idx):
        key = self.keys[idx]
        return self.loaded_data[key]

class RandomSubjectVisitDataset_V1(MultiIndependentVisitDataset_V1):
    def __init__(self, *args, seed=42, **kwargs):
        super().__init__(*args, **kwargs)
        self.N = len(self.subjects)
        self.rng = np.random.default_rng(seed=seed)

    def __getitem__(self, idx):
        sub = self.subjects[idx]
        df = self.df
        sub_df = df[df['sub'] == sub]
        visit = self.rng.choice(sub_df[self.visit_key].values)
        print(sub, visit)
        
        key = f'{sub}_{visit}'
        imgs_df = sub_df[sub_df['key'] == key]
        data_dict = self._load_imgsdf(imgs_df)
        
        return data_dict

class RandomSubjectVisitDatasetInRam_V1(MultiIndependentVisitDatasetInRam_V1):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.N = len(self.subjects)
        self.rng = np.random.default_rng(seed=seed)
        
    def __getitem__(self, idx):
        sub = self.subjects[idx]
        df = self.df
        sub_df = df[df['sub'] == sub]
        visit = self.rng.choice(sub_df[self.visit_key].values)
        print(sub, visit)
        
        key = f'{sub}_{visit}'
        return self.loaded_data[key]

