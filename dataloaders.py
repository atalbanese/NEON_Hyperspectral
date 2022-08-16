from msilib.schema import Error
from select import select
import h5_helper as hp
import warnings
#import pylas
from torch.utils.data import Dataset
import os
import random
import matplotlib.pyplot as plt
from skimage.segmentation import slic
import torch
import numpy as np
import utils
import h5py
from sklearn.preprocessing import StandardScaler
import transforms as tr
import torch.nn.functional as F
import torchvision.transforms as tt
import torchvision.transforms.functional as ttf
from einops import rearrange, reduce, repeat
import rasterio as rs
from rasterio import logging
from validator import Validator
from torch.utils.data import DataLoader
from tqdm import tqdm
from numpy.random import default_rng

import numpy.ma as ma
import pickle
from einops.layers.torch import Rearrange, Reduce
from attrs import define, field
#from torch_geometric.data import Data


class RenderBlocks(Dataset):
    def __init__(self,
                block_size: int,
                nmf_dir: str,
                save_dir: str,
                validator: Validator,
                key: str):
        self.block_size = block_size
        self.nmf_dir = nmf_dir
        self.save_dir = save_dir
        self.validator = validator
        self.key = key
        
        self.nmf_files = [os.path.join(self.nmf_dir, f) for f in os.listdir(self.nmf_dir) if ".npy" in f]

        def make_dict(file_list, param_1, param_2):
            return {f"{f.split('_')[param_1]}_{f.split('_')[param_2]}": f for f in file_list}

        self.nmf_dict = make_dict(self.nmf_files, -4, -3)
        self.all_files = list(set(self.nmf_dict.keys()) - set(self.validator.valid_files.keys()))

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, ix):
        key = self.all_files[ix]

        img = np.load(self.nmf_dict[key]).astype(np.float32)

        img = rearrange(img, '(h b1) (w b2) c -> (b1 b2) c h w', h = self.block_size, w=self.block_size)

        
        #it = np.nditer(img, flags=['f_index'])
        for index, x in enumerate(img):
            nan_sum = (x != x).sum()
            if nan_sum == 0:
                save_name = f'{key}_{index}.pt'
                nmf_tensor = torch.from_numpy(x)
                to_save = {self.key: nmf_tensor}
                save_loc = os.path.join(self.save_dir, save_name)
                with open(save_loc, 'wb') as f:
                    torch.save(to_save, save_loc)
        
        return 1


class RenderedDataLoader(Dataset):
    def __init__(self,
                file_folder,
                features,
                input_size = 20,
                stats_loc='',
                full_plots=False,
                scaling=True):
        self.scaling = scaling
        self.full_plots = full_plots
        if full_plots:
            self.crop = tt.RandomCrop(input_size)
        self.base_dir = file_folder
        self.files = os.listdir(file_folder)
        self.features = features
        if scaling:
            for ix, f in enumerate(self.files):
                if os.path.isdir(os.path.join(self.base_dir, f)):
                    del self.files[ix]
            self.calc_stats = True
            if os.path.exists(os.path.join(self.base_dir, 'stats/stats.npy')):
                self.stats = torch.load(os.path.join(self.base_dir, 'stats/stats.npy'))
                self.calc_stats = False
            elif os.path.exists(stats_loc):
                self.stats = torch.load(stats_loc)
                self.calc_stats = False
            self.scale_dict = {}
            
            for feature in features.keys():
                self.scale_dict[feature] = StandardScaler()
                if not self.calc_stats:
                    cur_stats = self.stats[feature]
                    self.scale_dict[feature].scale_ = cur_stats['scale']
                    self.scale_dict[feature].mean_ = cur_stats['mean']
                    self.scale_dict[feature].var_ = cur_stats['var']
            
            self.ra_1 = Rearrange('c h w -> (h w) c')
            #self.ra_2 = Rearrange('(h w) c -> c h w', h=input_size, w=input_size)

            self.flat_1 = Rearrange('h w -> (h w) ()')
            self.flat_2 = Rearrange('(h w) () -> h w', h=input_size, w=input_size)

    
    def save_stats(self):
        to_save = {}
        for key, value in self.scale_dict.items():
            to_save[key] = {'scale': value.scale_, 'mean': value.mean_, 'var': value.var_}

        save_loc = os.path.join(self.base_dir, 'stats/stats.npy')
        with open(save_loc, 'wb') as f:
            torch.save(to_save, save_loc)
        


    def __len__(self):
        return len(self.files)

    #Filter items with 0
    def __getitem__(self, ix):
        to_open = self.files[ix]
        

        to_return = torch.load(os.path.join(self.base_dir, to_open))

        if self.scaling:

            for key in to_return.keys():
                if key in self.features.keys():
                    scaler = self.scale_dict[key]
                    to_scale = to_return[key]
                    three_d = len(to_scale.shape) > 2
                    if three_d:
                        to_scale = self.ra_1(to_scale)
                    else:
                        to_scale = self.flat_1(to_scale)
                    if self.calc_stats:
                        scaler.partial_fit(to_scale)
                    to_scale = torch.from_numpy(scaler.transform(to_scale)).float()
                    if three_d:
                        to_scale = rearrange(to_scale, '(h w) c -> c h w', h=int(np.sqrt(to_scale.shape[0])), w=int(np.sqrt(to_scale.shape[0])))
                    else:
                        to_scale = self.flat_2(to_scale)
                    to_return[key] = to_scale
        else:
            to_scale = to_return[list(self.features.keys())[0]]
        
        if self.full_plots and to_scale.shape[2] != 20:
            params = self.crop.get_params(to_scale, (20, 20))
            for k, v in to_return.items():
                to_return[k] = ttf.crop(v, *params)

        return to_return
    






if __name__ == "__main__":


    

    NUM_CLASSES = 12
    NUM_CHANNELS = 10
    PCA_DIR= 'C:/Users/tonyt/Documents/Research/datasets/pca/niwo_masked_10'
    ICA_DIR = 'C:/Users/tonyt/Documents/Research/datasets/ica/niwo_10_channels'
    SHADOW_DIR ='C:/Users/tonyt/Documents/Research/datasets/mpsi/niwo'
    RAW_DIR = 'C:/Users/tonyt/Documents/Research/datasets/selected_bands/niwo/all'
   
    VALID_FILE = "W:/Classes/Research/neon-allsites-appidv-latest.csv"
    CURATED_FILE = "W:/Classes/Research/neon_niwo_mapped_struct.csv"
    PLOT_FILE = 'W:/Classes/Research/All_NEON_TOS_Plots_V8/All_NEON_TOS_Plots_V8/All_NEON_TOS_Plot_Centroids_V8.csv'

    CHM_DIR = 'C:/Users/tonyt/Documents/Research/datasets/chm/niwo/'
    AZM_DIR = 'C:/Users/tonyt/Documents/Research/datasets/solar_azimuth/niwo/'
    SP_DIR = 'C:/Users/tonyt/Documents/Research/datasets/superpixels/niwo'

    ORIG_DIR = 'W:/Classes/Research/datasets/hs/original/NEON.D13.NIWO.DP3.30006.001.2020-08.basic.20220516T164957Z.RELEASE-2022'
    SAVE_DIR = 'C:/Users/tonyt/Documents/Research/datasets/tensors/niwo_2020_pca_blocks/raw_training'
    INDEX_DIR = 'C:/Users/tonyt/Documents/Research/datasets/indexes/niwo'

    NMF_DIR = 'C:/Users/tonyt/Documents/Research/datasets/pca/niwo_16_unmasked/'

    CHM_MEAN = 4.015508459469479
    CHM_STD =  4.809300736115787

    test = RenderedDataLoader('C:/Users/tonyt/Documents/Research/datasets/tensors/niwo_2020_pca_blocks/full_plot_train', {'pca': 16}, stats_loc='C:/Users/tonyt/Documents/Research/datasets/tensors/niwo_2020_pca_blocks/raw_training/stats/stats.npy', full_plots=True)
    test.__getitem__(2)

    # test = MixingDataLoader('C:/Users/tonyt/Documents/Research/datasets/tensors/niwo_2020_hs_labels', {'PIEN': 0, 'ABLAL': 1, 'PIFL2': 2, 'PICOL': 3})
    # test.__getitem__(69)

    # valid = Validator(file=VALID_FILE, 
    #                 pca_dir=PCA_DIR, 
    #                 ica_dir=ICA_DIR,
    #                 raw_bands=RAW_DIR,
    #                 shadow=SHADOW_DIR,
    #                 site_name='NIWO', 
    #                 num_classes=NUM_CLASSES, 
    #                 plot_file=PLOT_FILE, 
    #                 struct=True, 
    #                 azm=AZM_DIR, 
    #                 chm=CHM_DIR, 
    #                 curated=CURATED_FILE, 
    #                 rescale=False, 
    #                 orig=ORIG_DIR, 
    #                 superpixel=SP_DIR,
    #                 indexes=INDEX_DIR,
    #                 prefix='D13',
    #                 chm_mean = 4.015508459469479,
    #                 chm_std = 4.809300736115787,
    #                 use_sp=True,
    #                 scholl_filter=False,
    #                 scholl_output=False,
    #                 filter_species = 'SALIX')

    #render = RenderDataLoader(PCA_DIR, CHM_DIR, AZM_DIR, SP_DIR, ICA_DIR, RAW_DIR, SHADOW_DIR, INDEX_DIR, 4.015508459469479, 4.809300736115787, 'raw_training', SAVE_DIR, validator=valid, patch_size=9)
    # render = RenderBlocks(20, NMF_DIR, SAVE_DIR, valid, 'pca')
    # #
    # train_loader = DataLoader(render, batch_size=1, num_workers=8)

    # for ix in tqdm(train_loader):
    #     1+1

    # rendered = RenderedDataLoader(SAVE_DIR, {'pca': 16}, input_size=20)
    # for ix in tqdm(rendered):
    #     1+1
    # rendered.save_stats()