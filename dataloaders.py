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
from einops import rearrange
import rasterio as rs
from rasterio import logging


import numpy.ma as ma
import pickle
from einops.layers.torch import Rearrange, Reduce
#from torch_geometric.data import Data

class MergedStructureDataset(Dataset):
    def __init__(self, pca_folder, tif_folder, azimuth_folder, superpix_folder, chm_mean, chm_std, crop_size=40, patch_size=4, sequence_length = 50, eval=False, rescale_pca=False, **kwargs):
        self.pca_folder = pca_folder
        self.chm_mean = chm_mean
        self.chm_std = chm_std
        self.crop_size = crop_size
        self.patch_size = patch_size
        self.superpix = superpix_folder
        self.sequence_length = sequence_length
        self.rescale_pca = rescale_pca
        self.batch_size = kwargs["batch_size"] if "batch_size" in kwargs else 32
        if os.path.exists(os.path.join(self.pca_folder, 'stats/good_files_dc.pkl')):
            with open(os.path.join(self.pca_folder, 'stats/good_files_dc.pkl'), 'rb') as f:
                self.files = pickle.load(f)
        else:
            self.files = [os.path.join(pca_folder,file) for file in os.listdir(pca_folder) if ".npy" in file]
            if not eval:
                self.check_files()
        self.rng = np.random.default_rng()

        def make_dict(file_list, param_1, param_2):
            return {(file.split('_')[param_1], file.split('_')[param_2]): file for file in file_list}

        self.chm_files = [os.path.join(tif_folder, file) for file in os.listdir(tif_folder) if ".tif" in file]
        self.azimuth_files = [os.path.join(azimuth_folder, file) for file in os.listdir(azimuth_folder) if ".npy" in file]
        self.superpix_files = [os.path.join(superpix_folder, file) for file in os.listdir(superpix_folder) if ".npy" in file]

        self.files_dict = make_dict(self.files, -4, -3)
        self.chm_dict = make_dict(self.chm_files, -3, -2)
        self.azimuth_dict = make_dict(self.azimuth_files, -4, -3)
        self.superpix_dict = make_dict(self.superpix_files, -4, -3)

        self.all_files = list(set(self.files_dict.keys()) & set(self.chm_dict.keys()) & set(self.azimuth_dict.keys()))


    def check_files(self):
        to_remove = []
        for file in self.files:
            try:
                img = np.load(file)
                img = rearrange(img, 'h w c -> (h w) c')
            except ValueError as e:
                to_remove.append(file)
            img = ma.masked_invalid(img)
            img = ma.compress_rows(img)
            if img.shape[0] <(1000*1000)/2:
                to_remove.append(file)
        self.files = list(set(self.files) - set(to_remove))
        with open(os.path.join(self.pca_folder, 'stats/good_files_dc.pkl'), 'wb') as f:
            pickle.dump(self.files, f)


    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, index):
        key = self.all_files[index]

        #PCA
        img = np.load(self.files_dict[key]).astype(np.float32)
        #img = rearrange(img, 'h w c -> (h w) c')
        img_mask = img == img
        #img = torch.from_numpy(img)

        


        #Azimuth
        azimuth = np.load(self.azimuth_dict[key]).astype(np.float32)
        #Make -1 to 1
        azimuth = (torch.from_numpy(azimuth)-180)/180
        azimuth[azimuth != azimuth] = 0
        #azimuth = np.ravel(azimuth)

        #CHM
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            chm_open = rs.open(self.chm_dict[key])
            chm = chm_open.read().astype(np.float32)
            chm[chm==-9999] = np.nan

        chm = (torch.from_numpy(chm).squeeze(0)- self.chm_mean)/self.chm_std
        chm[chm != chm] = 0
        #chm = np.ravel(chm)

        #Superpix
        #Masked segments are segment 0
        #superpixels = np.ravel(np.load(self.superpix_dict[key]))
        superpixels = np.load(self.superpix_dict[key])


        sp_values, sp_inverse = np.unique(superpixels, return_inverse=True)

        sp_select = self.rng.choice(range(1, sp_values.max()-1), size=sp_values.max()-2, replace=False)

        azms, chms, imgs, masks = [], [], [], []

        i = 0

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

            return arr[row_pad:row_len-row_pad-add_row, col_pad:col_len-col_pad-add_col, :]



        ra = Rearrange('h w c -> c h w')


        #TODO: augmentation is other random pix select
        #TODO: multi-crop strategy?

        while len(imgs) < self.batch_size:
            sequences = []
            while len(sequences) < self.sequence_length:
                pix_num = sp_select[i]
                select_length = sp_inverse == pix_num
                select_length = select_length.sum()
                if (select_length > 16):
                    crops = get_crop(superpixels, pix_num)
                    if (crops[1] - crops[0] >20) or (crops[3] - crops[2] >20):
                        i += 1
                        continue  
                    sp_crop = superpixels[crops[0]:crops[1], crops[2]:crops[3]] == pix_num
                    img_crop = img[crops[0]:crops[1], crops[2]:crops[3], :] * sp_crop[...,np.newaxis]
                    img_crop[img_crop != img_crop] =0
                    img_center = grab_center(img_crop, self.patch_size)
                    if img_center.shape == (4, 4, 10):
                        img_center = torch.from_numpy(img_center)
                        img_center = ra(img_center)
                        sequences.append(img_center)
                    i += 1
                else:
                    i += 1
            sqs = torch.stack(sequences)
            imgs.append(sqs)


             

        imgs = torch.stack(imgs)
        #masks = torch.stack(masks)
        #chms = torch.from_numpy(np.stack(chms))
        #azms = torch.from_numpy(np.stack(azms))

        to_return = {}
        to_return["base"] = imgs
        to_return['chm'] = chms
        to_return['azimuth'] = azms
        to_return['mask'] = masks

        return to_return

class MaskedStructureDataset(Dataset):
    def __init__(self, pca_folder, tif_folder, azimuth_folder, superpix_folder, crop_size, chm_mean, chm_std, eval=False, rescale_pca=False, **kwargs):
        self.pca_folder = pca_folder
        self.chm_mean = chm_mean
        self.chm_std = chm_std
        self.superpix = superpix_folder
        self.rescale_pca = rescale_pca
        self.batch_size = kwargs["batch_size"] if "batch_size" in kwargs else 256
        if os.path.exists(os.path.join(self.pca_folder, 'stats/good_files_dc.pkl')):
            with open(os.path.join(self.pca_folder, 'stats/good_files_dc.pkl'), 'rb') as f:
                self.files = pickle.load(f)
        else:
            self.files = [os.path.join(pca_folder,file) for file in os.listdir(pca_folder) if ".npy" in file]
            if not eval:
                self.check_files()
        self.rng = np.random.default_rng()
        self.crop_size = crop_size

        def make_dict(file_list, param_1, param_2):
            return {(file.split('_')[param_1], file.split('_')[param_2]): file for file in file_list}

        self.chm_files = [os.path.join(tif_folder, file) for file in os.listdir(tif_folder) if ".tif" in file]
        self.azimuth_files = [os.path.join(azimuth_folder, file) for file in os.listdir(azimuth_folder) if ".npy" in file]
        self.superpix_files = [os.path.join(superpix_folder, file) for file in os.listdir(superpix_folder) if ".npy" in file]

        self.files_dict = make_dict(self.files, -5, -4)
        self.chm_dict = make_dict(self.chm_files, -3, -2)
        self.azimuth_dict = make_dict(self.azimuth_files, -4, -3)
        self.superpix_dict = make_dict(self.superpix_files, -4, -3)

        self.all_files = list(set(self.files_dict.keys()) & set(self.chm_dict.keys()) & set(self.azimuth_dict.keys()))


    def check_files(self):
        to_remove = []
        for file in self.files:
            try:
                img = np.load(file)
                img = rearrange(img, 'h w c -> (h w) c')
            except ValueError as e:
                to_remove.append(file)
            img = ma.masked_invalid(img)
            img = ma.compress_rows(img)
            if img.shape[0] <(1000*1000)/2:
                to_remove.append(file)
        self.files = list(set(self.files) - set(to_remove))
        with open(os.path.join(self.pca_folder, 'stats/good_files_dc.pkl'), 'wb') as f:
            pickle.dump(self.files, f)


    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, index):
        key = self.all_files[index]

        #PCA
        img = np.load(self.files_dict[key]).astype(np.float32)
        img = rearrange(img, 'h w c -> (h w) c')
        img_mask = img == img
        #img = torch.from_numpy(img)

        


        #Azimuth
        azimuth = np.load(self.azimuth_dict[key]).astype(np.float32)
        #Make -1 to 1
        azimuth = (torch.from_numpy(azimuth)-180)/180
        azimuth[azimuth != azimuth] = 0
        azimuth = np.ravel(azimuth)

        #CHM
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            chm_open = rs.open(self.chm_dict[key])
            chm = chm_open.read().astype(np.float32)
            chm[chm==-9999] = np.nan

        chm = (torch.from_numpy(chm).squeeze(0)- self.chm_mean)/self.chm_std
        chm[chm != chm] = 0
        chm = np.ravel(chm)

        #Superpix
        #Masked segments are segment 0
        superpixels = np.ravel(np.load(self.superpix_dict[key])) * img_mask[:,0]

        sp_values, sp_index, sp_inverse = np.unique(superpixels, return_index=True, return_inverse=True)

        sp_select = self.rng.choice(range(1, sp_values.max()), size=self.batch_size, replace=False)

        azms, chms, imgs = [], [], []
        azms_mask, chms_mask, imgs_mask = [], [], []

        def select_and_mask(sm, arr, sl, slm):
            select = arr[sm]
            select_mask = np.ones_like(select)
            pad_mask = np.pad(select_mask, (0, 256- select_mask.shape[0]))
            pad = np.pad(select, (0, 256- select.shape[0]))
            sl.append(pad)
            slm.append(pad_mask)

        for ix, i in enumerate(sp_select):
            select_mask = sp_inverse == i
            
            select_and_mask(select_mask, azimuth, azms, azms_mask)
            select_and_mask(select_mask, chm, chms, chms_mask)
            img_select = img[select_mask, :]
            print(img_select.shape)
            img_mask = np.ones_like(img_select)
            img_pad = np.zeros((256 - img_select.shape[0], 10), dtype=img_select.dtype)
            img_select = np.concatenate((img_select, img_pad))
            img_mask = np.concatenate((img_mask, img_pad))
            imgs.append(img_select)
            imgs_mask.append(img_mask)

             

        to_return = {}
        to_return["base"] = img
        to_return['chm'] = chm
        to_return['azimuth'] = azimuth

        return to_return

class StructureDataset(Dataset):
    def __init__(self, pca_folder, tif_folder, azimuth_folder, crop_size, eval=False, rescale_pca=False, **kwargs):
        self.pca_folder = pca_folder
        self.rescale_pca = rescale_pca
        self.batch_size = kwargs["batch_size"] if "batch_size" in kwargs else 256
        if os.path.exists(os.path.join(self.pca_folder, 'stats/good_files_dc.pkl')):
            with open(os.path.join(self.pca_folder, 'stats/good_files_dc.pkl'), 'rb') as f:
                self.files = pickle.load(f)
        else:
            self.files = [os.path.join(pca_folder,file) for file in os.listdir(pca_folder) if ".npy" in file]
            if not eval:
                self.check_files()
        self.rng = np.random.default_rng()
        self.crop_size = crop_size

        def make_dict(file_list, param_1, param_2):
            return {(file.split('_')[param_1], file.split('_')[param_2]): file for file in file_list}

        self.chm_files = [os.path.join(tif_folder, file) for file in os.listdir(tif_folder) if ".tif" in file]
        self.azimuth_files = [os.path.join(azimuth_folder, file) for file in os.listdir(azimuth_folder) if ".npy" in file]

        self.files_dict = make_dict(self.files, -4, -3)
        self.chm_dict = make_dict(self.chm_files, -3, -2)
        self.azimuth_dict = make_dict(self.azimuth_files, -4, -3)

        self.all_files = list(set(self.files_dict.keys()) & set(self.chm_dict.keys()) & set(self.azimuth_dict.keys()))

       


    def check_files(self):
        to_remove = []
        for file in self.files:
            try:
                img = np.load(file)
                img = rearrange(img, 'h w c -> (h w) c')
            except ValueError as e:
                to_remove.append(file)
            img = ma.masked_invalid(img)
            img = ma.compress_rows(img)
            if img.shape[0] <1000*1000:
                to_remove.append(file)
        self.files = list(set(self.files) - set(to_remove))
        with open(os.path.join(self.pca_folder, 'stats/good_files_dc.pkl'), 'wb') as f:
            pickle.dump(self.files, f)



    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, index):
        key = self.all_files[index]

        #PCA
        img = np.load(self.files_dict[key]).astype(np.float32)
        img = rearrange(img, 'h w c -> c h w')
        img = torch.from_numpy(img)

        #Testing rescaling 0 to 1
        #Don't know if this will actually work with PCA, since its not image data but w/e 
        if self.rescale_pca:
            pca_min = -7.15022986754737
            pca_max = 18.434569044781508

            img = img-pca_min
            img /= (pca_max-pca_min)


        #Azimuth
        azimuth = np.load(self.azimuth_dict[key]).astype(np.float32)
        #Make -1 to 1
        azimuth = (torch.from_numpy(azimuth)-180)/180
        azimuth[azimuth != azimuth] = 0

        #CHM
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            chm_open = rs.open(self.chm_dict[key])
            chm = chm_open.read().astype(np.float32)
            chm[chm==-9999] = np.nan
        #HARV
        #Mean =  15.696561055743224
        #Std = 9.548285574843716

        #ABBY
        #Mean = 14.399022964154588
        #STD = 13.149885125626438

        #NIWO
        #MEAN = 4.015508459469479
        #std = 4.809300736115787
        chm = (torch.from_numpy(chm).squeeze(0)- 4.015508459469479)/4.809300736115787
        chm[chm != chm] = 0
             

        img = rearrange(img, 'c (b1 h) (b2 w) -> (b1 b2) c h w', h=self.crop_size, w=self.crop_size)
        azimuth = rearrange(azimuth, '(b1 h) (b2 w) -> (b1 b2) h w', h=self.crop_size, w=self.crop_size)
        chm = rearrange(chm, '(b1 h) (b2 w) -> (b1 b2) h w', h=self.crop_size, w=self.crop_size)

        random_select = self.rng.choice(range(0, img.shape[0]), size=self.batch_size, replace=False)
        img = img[random_select]
        chm = chm[random_select]
        azimuth = azimuth[random_select]

        to_return = {}
        to_return["base"] = img
        to_return['chm'] = chm
        to_return['azimuth'] = azimuth

        return to_return



class DeepClusterDataset(Dataset):
    def __init__(self, pca_folder, crop_size, eval=False, **kwargs):
        self.pca_folder = pca_folder
        self.batch_size = kwargs["batch_size"] if "batch_size" in kwargs else 128
        if os.path.exists(os.path.join(self.pca_folder, 'stats/good_files_dc.pkl')):
            with open(os.path.join(self.pca_folder, 'stats/good_files_dc.pkl'), 'rb') as f:
                self.files = pickle.load(f)
        else:
            self.files = [os.path.join(pca_folder,file) for file in os.listdir(pca_folder) if ".npy" in file]
            if not eval:
                self.check_files()
        self.rng = np.random.default_rng()
        self.crop_size = crop_size

       

    def check_files(self):
        to_remove = []
        for file in self.files:
            try:
                img = np.load(file)
                img = rearrange(img, 'h w c -> (h w) c')
            except ValueError as e:
                to_remove.append(file)
            img = ma.masked_invalid(img)
            img = ma.compress_rows(img)
            if img.shape[0] <1000*1000:
                to_remove.append(file)
        self.files = list(set(self.files) - set(to_remove))
        with open(os.path.join(self.pca_folder, 'stats/good_files_dc.pkl'), 'wb') as f:
            pickle.dump(self.files, f)



    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        img = np.load(self.files[index]).astype(np.float32)
        img = rearrange(img, 'h w c -> c h w')
        img = torch.from_numpy(img)     

        img = rearrange(img, 'c (b1 h) (b2 w) -> (b1 b2) c h w', h=self.crop_size, w=self.crop_size)
        random_select = self.rng.choice(range(0, img.shape[0]), size=self.batch_size, replace=False)
        img = img[random_select]

        to_return = {}
        to_return["base"] = img

        return to_return

class MaskedDenseVitDataset(Dataset):
    def __init__(self, pca_folder, crop_size, eval=False, **kwargs):
        self.pca_folder = pca_folder
        self.batch_size = kwargs["batch_size"] if "batch_size" in kwargs else 64
        # self.mean = np.load(os.path.join(pca_folder, 'stats/mean.npy')).astype(np.float64)
        # self.std = np.load(os.path.join(pca_folder, 'stats/std.npy')).astype(np.float64)
        # self.norm = tt.Normalize(self.mean, self.std)
        if os.path.exists(os.path.join(self.pca_folder, 'stats/good_files.pkl')):
            with open(os.path.join(self.pca_folder, 'stats/good_files.pkl'), 'rb') as f:
                self.files = pickle.load(f)
        else:
            self.files = [os.path.join(pca_folder,file) for file in os.listdir(pca_folder) if ".npy" in file]
            if not eval:
                self.check_files()
        self.rng = np.random.default_rng()
        self.crop_size = crop_size

        self.random_crop = tt.RandomCrop(crop_size)

        self.transforms_1 = tt.Compose([tt.RandomHorizontalFlip(),
                                    tt.RandomVerticalFlip()])
        # self.transforms_2 = tt.Compose([
        #                             tr.RandomPointMaskEven(),
        #                             tr.RandomRectangleMaskEven()])

        self.blit = tr.Blit()
        self.block = tr.Block()
        self.make_linear = Rearrange('b c h w -> b (h w) c')
        self.make_2d = Rearrange('b (h w) c -> b c h w', h=self.crop_size, w = self.crop_size)
        

    def check_files(self):
        to_remove = []
        for file in self.files:
            try:
                img = np.load(file)
                img = rearrange(img, 'h w c -> (h w) c')
            except ValueError as e:
                to_remove.append(file)
            img = ma.masked_invalid(img)
            img = ma.compress_rows(img)
            if img.shape[0] < 800*800:
                to_remove.append(file)
        self.files = list(set(self.files) - set(to_remove))
        with open(os.path.join(self.pca_folder, 'stats/good_files.pkl'), 'wb') as f:
            pickle.dump(self.files, f)



    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        img = np.load(self.files[index]).astype(np.float32)
        img = rearrange(img, 'h w c -> c h w')
        img = torch.from_numpy(img).unsqueeze(0)
        img = f.interpolate(img, size=(1024,1024))
        img = img.squeeze()
        

        #img = self.random_crop(img)
        mask = img != img
        #Set nans to 0
        img[mask] = 0
        #img = self.norm(img)
        #Reset them to 0 as they will be 'normalized' now
        img[mask] = 0
        img = rearrange(img, 'c (b1 h) (b2 w) -> (b1 b2) c h w', h=self.crop_size, w=self.crop_size)
        mask = rearrange(mask, 'c (b1 h) (b2 w) -> (b1 b2) c h w', h=self.crop_size, w=self.crop_size)
        random_select = self.rng.choice(range(0, img.shape[0]), size=self.batch_size, replace=False)
        img = img[random_select]
        mask = mask[random_select]


        to_return = {}
        to_return["base"] = self.transforms_1(img)
        augment = self.make_linear(img)
        augment = self.blit(self.block(augment))
        augment = self.make_2d(augment)
        to_return['augment'] = augment
        to_return["mask"] = mask
        #to_return["full"] = full

        return to_return

class MaskedVitDataset(Dataset):
    def __init__(self, pca_folder, crop_size, eval=False, **kwargs):
        self.batch_size = kwargs["batch_size"] if "batch_size" in kwargs else 128
        self.mean = np.load(os.path.join(pca_folder, 'stats/mean.npy')).astype(np.float64)
        self.std = np.load(os.path.join(pca_folder, 'stats/std.npy')).astype(np.float64)
        self.norm = tt.Normalize(self.mean, self.std)
        self.files = [os.path.join(pca_folder,file) for file in os.listdir(pca_folder) if ".npy" in file]
        self.rng = np.random.default_rng()
        self.crop_size = crop_size

        self.transforms_1 = tt.Compose([tt.RandomHorizontalFlip(),
                                    tt.RandomVerticalFlip()])
        self.transforms_2 = tt.Compose([
                                    tr.RandomPointMask(),
                                    tr.RandomRectangleMask()])

        if not eval:
            self.check_files()

    def check_files(self):
        to_remove = []
        for file in self.files:
            img = np.load(file)
            img = rearrange(img, 'h w c -> (h w) c')
            img = ma.masked_invalid(img)
            img = ma.compress_rows(img)
            if img.shape[0] < self.batch_size * 4:
                to_remove.append(file)
        self.files = list(set(self.files) - set(to_remove))


    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        img = np.load(self.files[index]).astype(np.float32)
        img = torch.from_numpy(img)
        #img = rearrange(img, 'h w c -> c h w')
        mask = img != img
        #Set nans to 0
        img[mask] = 0
        img = self.norm(img)
        #Reset them to 0 as they will be 'normalized' now
        img[mask] = 0
        img = rearrange(img, 'c (b1 h) (b2 w) -> (b1 b2) c h w', h=self.crop_size, w=self.crop_size)
        mask = rearrange(mask, 'c (b1 h) (b2 w) -> (b1 b2) c h w', h=self.crop_size, w=self.crop_size)
        random_select = self.rng.choice(range(0, img.shape[0]), size=self.batch_size, replace=False)
        img = img[random_select]
        mask = mask[random_select]

        to_return = {}
        to_return["base"] = self.transforms_1(img)
        to_return["augment"] = self.transforms_2(to_return["base"])
        to_return["mask"] = mask
        #to_return["full"] = full

        return to_return



class MaskedDataset(Dataset):
    def __init__(self, pca_folder, **kwargs):
        self.batch_size = kwargs["batch_size"] if "batch_size" in kwargs else 128
        self.mean = np.load(os.path.join(pca_folder, 'stats/mean.npy')).astype(np.float32)
        self.std = np.load(os.path.join(pca_folder, 'stats/std.npy')).astype(np.float32)
        self.files = [os.path.join(pca_folder,file) for file in os.listdir(pca_folder) if ".npy" in file]
        self.rng = np.random.default_rng()

        self.flip = tr.Flip()
        self.blit = tr.Blit()
        self.blit_darken = tr.BlitDarken()
        self.block = tr.Block()

        self.check_files()

    def check_files(self):
        to_remove = []
        for file in self.files:
            img = np.load(file)
            img = rearrange(img, 'h w c -> (h w) c')
            img = ma.masked_invalid(img)
            img = ma.compress_rows(img)
            if img.shape[0] < self.batch_size * 4:
                to_remove.append(file)
        self.files = list(set(self.files) - set(to_remove))


    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        to_return = {}
        img = np.load(self.files[index]).astype(np.float32)

        img = rearrange(img, 'h w c -> (h w) c')
        #to_return['orig_shape'] = img.shape
        img = ma.masked_invalid(img)
        #to_return['mask'] = img.mask
        img = ma.compress_rows(img)

        random_select = self.rng.choice(range(0, img.shape[0]), size=self.batch_size, replace=False)
        img = img[random_select]
        img = (img - self.mean)/self.std
        to_return['base'] = self.flip(img).copy()
        to_return['augment'] = self.block(self.blit_darken(self.blit(img)))

        return to_return

    

         

        





class PreProcDataset(Dataset):
    def __init__(self, pca_folder, **kwargs):
        self.rearrange = kwargs['rearrange'] if 'rearrange' in kwargs else False
        self.batch_size = kwargs["batch_size"] if "batch_size" in kwargs else 128
        self.crop_size = kwargs["crop_size"] if "crop_size" in kwargs else 25
        self.mean = np.load(os.path.join(pca_folder, 'stats/mean.npy')).astype(np.float64)
        self.std = np.load(os.path.join(pca_folder, 'stats/std.npy')).astype(np.float64)
        self.norm = tt.Normalize(self.mean, self.std)
        self.transforms_1 = tt.Compose([tt.RandomHorizontalFlip(),
                                    tt.RandomVerticalFlip()])
        self.transforms_2 = tt.Compose([
                                    tr.RandomPointMask(),
                                    tr.RandomRectangleMask()])
        
        self.files = [os.path.join(pca_folder,file) for file in os.listdir(pca_folder) if ".npy" in file]
        self.rng = np.random.default_rng()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        img = np.load(self.files[index]).astype(np.float32)
        img = torch.from_numpy(img)
        if self.rearrange:
            img = rearrange(img, 'h w c -> c h w')
        img = self.norm(img)
        full = img
        img = rearrange(img, 'c (b1 h) (b2 w) -> (b1 b2) c h w', h=self.crop_size, w=self.crop_size)
        img[img != img] = -10
        random_select = self.rng.choice(range(0, img.shape[0]), size=self.batch_size, replace=False)
        img = img[random_select]

        to_return = {}
        to_return["base"] = self.transforms_1(img)
        to_return["rand"] = self.transforms_2(to_return["base"])
        to_return["full"] = full

        return to_return

        

class HyperDataset(Dataset):
    def __init__(self, hyper_folder, **kwargs):
        self.viz_bands = {"red": 654,
                        "green": 561, 
                        "blue": 482}
        self.kwargs = kwargs
        self.rng = np.random.default_rng()
        #self.pca = kwargs["pca"] if "pca" in kwargs else True
        self.augment_type = kwargs["augment"] if "augment" in kwargs else "wavelength"
        self.num_bands = kwargs["num_bands"] if "num_bands" in kwargs else 30
        self.h5_location = hyper_folder
        self.batch_size = kwargs["batch_size"] if "batch_size" in kwargs else 32
        self.crop_size = kwargs["crop_size"] if "crop_size" in kwargs else 27
        self.transforms_1 = tt.Compose([tt.RandomHorizontalFlip(),
                                    tt.RandomVerticalFlip()])
        self.transforms_2 = tt.Compose([
                                    tr.RandomPointMask(),
                                    tr.RandomRectangleMask()])
                                    
        h5_files = [file for file in os.listdir(self.h5_location) if ".h5" in file]
        
        def make_dict(file_list, param_1, param_2):
            return {(file.split('_')[param_1], file.split('_')[param_2]): file for file in file_list}

        self.h5_dict = make_dict(h5_files, -3, -2)

        self.files = list(self.h5_dict.keys())

        #TODO: Test mode to enable/disable this
        self.clear_nans()
    


    def clear_nans(self):
        band = {'b1': 500}
        count = 0
        to_remove = []
        for key, value in self.h5_dict.items():
            bands = hp.pre_processing(os.path.join(self.h5_location, value), wavelength_ranges=band)
            if np.isnan(bands["bands"]['b1']).sum() > (1000*1000*.10):
                to_remove.append(key)
                count += 1

        for remove in to_remove:
            del self.h5_dict[remove]
            self.files.remove(remove)

        print(f"removed {count} files from file list since they were >10% missing values")


    def make_crops(self):
        #There is definitely a faster way to do this but it works
        crops = []
        while len(crops) < self.batch_size:
            x_min, y_min = random.randint(0,1000-self.crop_size), random.randint(0,1000-self.crop_size)
            x_max, y_max = x_min + self.crop_size, y_min + self.crop_size
            crop_dims = (x_min, y_min, x_max, y_max)
            x_range = range(x_min, x_max)
            y_range = range(y_min, y_max)
            for crop in crops:
                crop_x_range = range(crop[0], crop[2])
                crop_y_range = range(crop[1], crop[3])
                x_int = list(set(x_range) & set(crop_x_range))
                y_int = list(set(y_range) & set(crop_y_range))
                if len(x_int) * len(y_int):
                    break
            else:
                crops.append(crop_dims)
        return crops

    def make_crops_det(self):
        crops = []
        start = random.randint(0, self.crop_size-1)
        for i in range(start, 1000-self.crop_size, self.crop_size):
            min_x = i
            max_x = i + self.crop_size
            for j in range(start, 1000-self.crop_size, self.crop_size):
                min_y = j
                max_y = j +self.crop_size
                crops.append((min_x, min_y, max_x, max_y))
        crops = random.sample(crops, k =self.batch_size)
        return crops

    # Just for debugging
    def plot_crops(self):
        crops = self.make_crops()
        for crop in crops:
            x_list = [crop[0], crop[2], crop[2], crop[0], crop[0]]
            y_list = [crop[1], crop[1], crop[3], crop[3], crop[1]]
            plt.plot(x_list, y_list)
        plt.show()

    def make_h5_stack(self, h5, crops):
        if isinstance(h5, dict):
            h5 = hp.stack_all(h5, axis=0)

        h5_samples = [h5[:, crop[0]:crop[2], crop[1]:crop[3]] for crop in crops]
        # Convert any NAN values to -1
        h5_tensor_list = []
        for sample in h5_samples:
            sample[sample != sample] = -1
            sample = sample.astype(np.float32)
            h5_tensor_list.append(torch.from_numpy(sample))

        return torch.stack(h5_tensor_list)


    def random_band_select(self, selected):
        #TODO: fix magic number here
        possible_bands = set(range(0, 423))
        selected = set([value for value in selected.values()])
        to_select = list(possible_bands - selected)
        return self.rng.choice(to_select, size=self.num_bands, replace=False)

    def semi_rand_band_select(self, selected, width=5):

        selected = [value for value in selected.values()]
        change_window = list(range(-width, width))
        changes = self.rng.choice(change_window, size=self.num_bands)
        return [selected[i] + changes[i] for i in range(0,self.num_bands)]
    
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        to_return = {}
        coords = self.files[idx]
        h5 = self.h5_dict[coords]
        f = h5py.File(os.path.join(self.h5_location, h5))
        all_data = hp.pre_processing(f, get_all=True)
        base_h5 = utils.pca(all_data["bands"], True, n_components=self.num_bands, whiten=False)
        to_return["base"] = base_h5
        viz_h5= hp.pre_processing(f, wavelength_ranges=self.viz_bands)
        to_return["viz"] = viz_h5["bands"]

        #Deprecated - random wavelength augmentation, switching to PCA based augmentation
        # if self.augment_type == "wavelength":
        #     random_bands = self.random_band_select(selected)
        #     #random_bands = self.semi_rand_band_select(selected)
        #     random_bands = {i: random_bands[i] for i in range(0, len(random_bands))}

        #     rand_h5, _, _ = self.process_h5(f, select_bands=random_bands)
        #     to_return["rand"] = rand_h5

        
        #crops = self.make_crops()
        crops = self.make_crops_det()

        f.close()

        to_return = {key: self.make_h5_stack(value, crops) for key, value in to_return.items()}
        to_return["base"] = self.transforms_1(to_return["base"])
        to_return["rand"] = self.transforms_2(to_return["base"])
        #DOUBLE AUGMENT
        #to_return["base"] = self.transforms_1(to_return["base"])

        return to_return
    


if __name__ == "__main__":

    pca_fold = 'C:/Users/tonyt/Documents/Research/datasets/pca/niwo_masked_10'
    chm_fold = 'C:/Users/tonyt/Documents/Research/datasets/chm/niwo'
    az_fold = 'C:/Users/tonyt/Documents/Research/datasets/solar_azimuth/niwo'
    sp_fold = 'C:/Users/tonyt/Documents/Research/datasets/superpixels/niwo'

    test = MergedStructureDataset(pca_fold, chm_fold, az_fold, sp_fold, 4.015508459469479, 4.809300736115787, eval=True)
    # test = MaskedDenseVitDataset(pca_fold, 8, eval=True)

    print(test.__getitem__(69).shape)