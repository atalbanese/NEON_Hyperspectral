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
import transforms as tr
import torch.nn.functional as f
import torchvision.transforms as tt
from einops import rearrange, reduce
import rasterio as rs
from rasterio import logging
from validator import Validator
from tqdm import tqdm


import numpy.ma as ma
import pickle
from einops.layers.torch import Rearrange, Reduce
#from torch_geometric.data import Data



class RenderWholePixDataLoader(Dataset):
    def __init__(self, 
                    pca_folder, 
                    tif_folder, 
                    azimuth_folder, 
                    superpix_folder, 
                    chm_mean, 
                    chm_std, 
                    mode,
                    save_dir,
                    patch_size=4, 
                    validator=None, 
                    **kwargs):
        self.pca_folder = pca_folder
        self.chm_mean = chm_mean
        self.chm_std = chm_std
        self.patch_size = patch_size
        self.superpix = superpix_folder
        self.validator = validator
        self.mode = mode
        self.save_dir = save_dir
        self.resize = 16
        self.files = [os.path.join(pca_folder,file) for file in os.listdir(pca_folder) if ".npy" in file]
        self.rng = np.random.default_rng()

        def make_dict(file_list, param_1, param_2):
            return {f"{file.split('_')[param_1]}_{file.split('_')[param_2]}": file for file in file_list}

        self.chm_files = [os.path.join(tif_folder, file) for file in os.listdir(tif_folder) if ".tif" in file]
        self.azimuth_files = [os.path.join(azimuth_folder, file) for file in os.listdir(azimuth_folder) if ".npy" in file]
        self.superpix_files = [os.path.join(superpix_folder, file) for file in os.listdir(superpix_folder) if ".npy" in file]

        self.files_dict = make_dict(self.files, -4, -3)
        self.chm_dict = make_dict(self.chm_files, -3, -2)
        self.azimuth_dict = make_dict(self.azimuth_files, -4, -3)
        self.superpix_dict = make_dict(self.superpix_files, -4, -3)

        self.all_files = set(self.files_dict.keys()) & set(self.chm_dict.keys()) & set(self.azimuth_dict.keys()) & set(self.superpix_dict.keys())

        self.filter_files()

    def filter_files(self):
        valid_keys = set(self.validator.valid_files.keys())
        if self.mode == 'raw_training':
            self.all_files = list(self.all_files - valid_keys)
        else: 
            self.all_files = self.valid_keys

        return None

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, index):
        key = self.all_files[index]

        #PCA
        img = np.load(self.files_dict[key]).astype(np.float32)
     
        #Azimuth
        azm = np.load(self.azimuth_dict[key]).astype(np.float32)
        azm = (azm-180)/180
        azm[azm != azm] = 0

        #CHM
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            chm_open = rs.open(self.chm_dict[key])
            chm = chm_open.read().astype(np.float32)
            chm[chm==-9999] = np.nan

        chm = (chm.squeeze(axis=0)- self.chm_mean)/self.chm_std
        chm[chm != chm] = 0

        #Superpix
        #Masked segments are segment 0
        #superpixels = np.ravel(np.load(self.superpix_dict[key]))
        superpixels = np.load(self.superpix_dict[key])

        sp_values, sp_inverse = np.unique(superpixels, return_inverse=True)

        def get_crop(arr, ix):
            locs = arr != ix
            falsecols = np.all(locs, axis=0)
            falserows = np.all(locs, axis=1)

            firstcol = falsecols.argmin()
            firstrow = falserows.argmin()

            lastcol = len(falsecols) - falsecols[::-1].argmin()
            lastrow = len(falserows) - falserows[::-1].argmin()

            return firstrow,lastrow,firstcol,lastcol
        
        def get_pad(arr, pad_size):
            row_len = arr.shape[0]
            col_len = arr.shape[1]

            row_pad = (pad_size - row_len) // 2
            col_pad = (pad_size - col_len) // 2
            
            add_row = (row_pad*2 + row_len) != pad_size
            add_col = (col_pad*2 + col_len) != pad_size

            return [(row_pad, row_pad+add_row), (col_pad, col_pad+add_col)]

        def grab_center(arr, diam):
            row_len = arr.shape[0]
            col_len = arr.shape[1]

            row_pad = (row_len - diam) // 2
            col_pad = (col_len - diam) // 2

            add_row = (row_pad * 2) + diam != row_len
            add_col = (col_pad * 2) + diam != col_len

            return arr[row_pad:row_len-row_pad-add_row, col_pad:col_len-col_pad-add_col,...]

        ra = Rearrange('h w c -> c h w')

        for pix_num in sp_values[1:]:
            select_length = sp_inverse == pix_num
            select_length = select_length.sum()
            if (select_length > 16):
                crops = get_crop(superpixels, pix_num)
                if (crops[1] - crops[0] >self.resize) or (crops[3] - crops[2] >self.resize):
                    continue
                if (crops[1] - crops[0] <3) or (crops[3] - crops[2]<3):
                    continue
                sp_crop = superpixels[crops[0]:crops[1], crops[2]:crops[3]] == pix_num
                img_crop = img[crops[0]:crops[1], crops[2]:crops[3], :] * sp_crop[...,np.newaxis]
                if img_crop.sum() == 0:
                    continue
                mask = img_crop != img_crop
                img_crop[mask] = 0
                mask = reduce(mask, 'h w c-> () h w', 'max').squeeze(0)


                chm_crop = chm[crops[0]:crops[1], crops[2]:crops[3]] * sp_crop
                if chm_crop.sum() == 0:
                    continue
                chm_mask = chm_crop != chm_crop
                chm_crop[chm_mask] = 0

                azm_crop = azm[crops[0]:crops[1], crops[2]:crops[3]] * sp_crop
                azm_mask = azm_crop != azm_crop
                azm_crop[azm_mask] = 0  

                mask += chm_mask
                mask += azm_mask

                mask = ~mask

                pad = get_pad(img_crop, self.resize)

                mask = np.pad(mask, pad)

                chm_center = np.pad(chm_crop, pad)
                azm_center = np.pad(azm_crop, pad)

                pad.append((0,0))
                img_center = np.pad(img_crop, pad)


                img_center = torch.from_numpy(img_center)
                img_center = ra(img_center)

                chm_center = torch.from_numpy(chm_center)

                azm_center = torch.from_numpy(azm_center)
                mask = torch.from_numpy(mask)

                to_save = {'img': img_center,
                            'azm': azm_center,
                            'chm': chm_center,
                            'mask': mask}

                f_name = f"coords_{key}_sp_index_{pix_num}.pt"

                save_loc = os.path.join(self.save_dir, f_name)
                with open(save_loc, 'wb') as f:
                    torch.save(to_save, save_loc)

        return None

class RenderedDataLoader(Dataset):
    def __init__(self,
                file_folder):
        self.base_dir = file_folder
        self.files = os.listdir(file_folder)

    def __len__(self):
        return len(self.files)

    #Filter items with 0
    def __getitem__(self, ix):
        to_open = self.files[ix]
        
        try:
            to_return = torch.load(os.path.join(self.base_dir, to_open))
        except:
            print(to_open)
        to_return['pca'][to_return['pca'] != to_return['pca']] = 0
        to_return['ica'][to_return['ica'] != to_return['ica']] = 0


        return to_return


class RenderDataLoader(Dataset):
    def __init__(self, 
                    pca_folder, 
                    tif_folder, 
                    azimuth_folder, 
                    superpix_folder,
                    ica_folder,
                    extra_bands_folder, 
                    shadow_folder,
                    chm_mean, 
                    chm_std, 
                    mode,
                    save_dir,
                    patch_size=4, 
                    validator=None, 
                    **kwargs):
        self.pca_folder = pca_folder
        self.chm_mean = chm_mean
        self.chm_std = chm_std
        self.patch_size = patch_size
        self.superpix = superpix_folder
        self.ica_folder = ica_folder
        self.extra_bands_folder = extra_bands_folder
        self.validator = validator
        self.mode = mode
        self.save_dir = save_dir
        self.resize = 16
        self.pca_files = [os.path.join(pca_folder,file) for file in os.listdir(pca_folder) if ".npy" in file]
        self.rng = np.random.default_rng()

        def make_dict(file_list, param_1, param_2):
            return {f"{file.split('_')[param_1]}_{file.split('_')[param_2]}": file for file in file_list}

        self.chm_files = [os.path.join(tif_folder, file) for file in os.listdir(tif_folder) if ".tif" in file]
        self.azimuth_files = [os.path.join(azimuth_folder, file) for file in os.listdir(azimuth_folder) if ".npy" in file]
        self.superpix_files = [os.path.join(superpix_folder, file) for file in os.listdir(superpix_folder) if ".npy" in file]
        self.ica_files = [os.path.join(ica_folder, file) for file in os.listdir(ica_folder) if ".npy" in file]
        self.extra_files = [os.path.join(extra_bands_folder, file) for file in os.listdir(extra_bands_folder) if ".npy" in file]
        self.shadow_files = [os.path.join(shadow_folder, file) for file in os.listdir(shadow_folder) if ".npy" in file]

        self.pca_files_dict = make_dict(self.pca_files, -4, -3)
        self.chm_dict = make_dict(self.chm_files, -3, -2)
        self.azimuth_dict = make_dict(self.azimuth_files, -4, -3)
        self.superpix_dict = make_dict(self.superpix_files, -4, -3)
        self.ica_files_dict = make_dict(self.ica_files, -5, -4)
        self.extra_files_dict = make_dict(self.extra_files, -4, -3)
        self.shadow_dict = make_dict(self.shadow_files, -4, -3)

        self.all_files = set(self.pca_files_dict.keys()) & set(self.chm_dict.keys()) & set(self.azimuth_dict.keys()) & set(self.superpix_dict.keys()) & set(self.ica_files_dict.keys()) & set(self.extra_files_dict.keys()) & set(self.shadow_dict.keys())

        self.filter_files()

    def filter_files(self):
        valid_keys = set(self.validator.valid_files.keys())
        if self.mode == 'raw_training':
            self.all_files = list(self.all_files - valid_keys)
        else: 
            self.all_files = self.valid_keys

        return None

    @staticmethod
    def get_crop(arr, ix):
            locs = arr != ix
            falsecols = np.all(locs, axis=0)
            falserows = np.all(locs, axis=1)

            firstcol = falsecols.argmin()
            firstrow = falserows.argmin()

            lastcol = len(falsecols) - falsecols[::-1].argmin()
            lastrow = len(falserows) - falserows[::-1].argmin()

            return firstrow,lastrow,firstcol,lastcol
    @staticmethod
    def grab_center(arr, diam):
            row_len = arr.shape[0]
            col_len = arr.shape[1]

            row_pad = (row_len - diam) // 2
            col_pad = (col_len - diam) // 2

            add_row = (row_pad * 2) + diam != row_len
            add_col = (col_pad * 2) + diam != col_len

            return arr[row_pad:row_len-row_pad-add_row, col_pad:col_len-col_pad-add_col,...]


    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, index):
        key = self.all_files[index]

        #PCA
        pca = np.load(self.pca_files_dict[key]).astype(np.float32)

        #ICA
        ica = np.load(self.ica_files_dict[key]).astype(np.float32)

        #Shadow Index
        shadow = np.load(self.shadow_dict[key]).astype(np.float32)

        #Raw bands
        extra = np.load(self.extra_files_dict[key]).astype(np.float32)
     
        #Azimuth
        azm = np.load(self.azimuth_dict[key]).astype(np.float32)
        azm = (azm-180)/180
        azm[azm != azm] = 0

        #CHM
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            chm_open = rs.open(self.chm_dict[key])
            chm = chm_open.read().astype(np.float32)
            chm[chm==-9999] = np.nan

        chm = chm.squeeze(axis=0)
        chm[chm != chm] = 0

        #Superpix
        #Masked segments are segment 0
        #superpixels = np.ravel(np.load(self.superpix_dict[key]))
        superpixels = np.load(self.superpix_dict[key])

        sp_values, sp_inverse = np.unique(superpixels, return_inverse=True)
  
        ra = Rearrange('h w c -> c h w')

        for pix_num in sp_values[1:]:
            select_length = sp_inverse == pix_num
            select_length = select_length.sum()
            if (select_length > 16):
                crops = self.get_crop(superpixels, pix_num)
                if (crops[1] - crops[0] >20) or (crops[3] - crops[2] >20):
                    continue
                sp_crop = superpixels[crops[0]:crops[1], crops[2]:crops[3]] == pix_num


                pca_crop = pca[crops[0]:crops[1], crops[2]:crops[3], :] * sp_crop[...,np.newaxis]
                pca_crop[pca_crop != pca_crop] = 0

                ica_crop = ica[crops[0]:crops[1], crops[2]:crops[3], :] * sp_crop[...,np.newaxis]
                ica_crop[ica_crop != ica_crop] = 0

                extra_crop = extra[crops[0]:crops[1], crops[2]:crops[3], :] * sp_crop[...,np.newaxis]
                extra_crop[extra_crop != extra_crop] = 0


                chm_crop = chm[crops[0]:crops[1], crops[2]:crops[3]] * sp_crop
                chm_crop[chm_crop != chm_crop] = 0

                azm_crop = azm[crops[0]:crops[1], crops[2]:crops[3]] * sp_crop
                azm_crop[azm_crop != azm_crop] = 0

                shadow_crop = shadow[crops[0]:crops[1], crops[2]:crops[3]] * sp_crop
                shadow_crop[shadow_crop != shadow_crop] = 0

                pca_center = self.grab_center(pca_crop, self.patch_size)
                if (pca_center.shape == (self.patch_size, self.patch_size, 10)) and (pca_center.sum() != 0):
                    pca_center = torch.from_numpy(pca_center)
                    pca_center = ra(pca_center)
                    
                    ica_center = self.grab_center(ica_crop, self.patch_size)
                    ica_center = torch.from_numpy(ica_center)
                    ica_center = ra(ica_center)

                    extra_center = self.grab_center(extra_crop, self.patch_size)
                    extra_center = torch.from_numpy(extra_center)
                    extra_center = ra(extra_center)

                    chm_center = self.grab_center(chm_crop, self.patch_size)
                    chm_center = torch.from_numpy(chm_center)

                    azm_center = self.grab_center(azm_crop, self.patch_size)
                    azm_center = torch.from_numpy(azm_center)

                    shadow_center = self.grab_center(shadow_crop, self.patch_size)
                    shadow_center = torch.from_numpy(shadow_center)

                    to_save = {'pca': pca_center,
                                'ica': ica_center,
                                'raw_bands': extra_center,
                                'shadow': shadow_center,
                                'azm': azm_center,
                                'chm': chm_center}

                    f_name = f"coords_{key}_sp_index_{pix_num}.pt"

                    save_loc = os.path.join(self.save_dir, f_name)
                    with open(save_loc, 'wb') as f:
                        torch.save(to_save, save_loc)

        return None



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
    SAVE_DIR = 'C:/Users/tonyt/Documents/Research/datasets/tensors/niwo_2020_pca_ica_shadow_extra_all/raw_training'


    CHM_MEAN = 4.015508459469479
    CHM_STD =  4.809300736115787

    valid = Validator(file=VALID_FILE, 
                    pca_dir=PCA_DIR, 
                    ica_dir=ICA_DIR,
                    raw_bands=RAW_DIR,
                    shadow=SHADOW_DIR,
                    site_name='NIWO', 
                    num_classes=NUM_CLASSES, 
                    plot_file=PLOT_FILE, 
                    struct=True, 
                    azm=AZM_DIR, 
                    chm=CHM_DIR, 
                    curated=CURATED_FILE, 
                    rescale=False, 
                    orig=ORIG_DIR, 
                    superpixel=SP_DIR,
                    prefix='D13',
                    chm_mean = 4.015508459469479,
                    chm_std = 4.809300736115787,
                    patch_size=3)

    render = RenderDataLoader(PCA_DIR, CHM_DIR, AZM_DIR, SP_DIR, ICA_DIR, RAW_DIR, SHADOW_DIR, 4.015508459469479, 4.809300736115787, 'raw_training', SAVE_DIR, validator=valid)
    # test = MaskedDenseVitDataset(pca_fold, 8, eval=True)

    for ix in tqdm(render):
        print(ix)