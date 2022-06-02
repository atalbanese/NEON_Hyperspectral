import h5_helper as hp
import warnings
#import pylas
from torch.utils.data import Dataset
import os
import random
import matplotlib.pyplot as plt
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

    pca_fold = 'C:/Users/tonyt/Documents/Research/datasets/pca/harv_2022_10_channels/'
    chm_fold = 'C:/Users/tonyt/Documents/Research/datasets/chm/harv_2019/NEON_struct-ecosystem/NEON.D01.HARV.DP3.30015.001.2019-08.basic.20220511T165943Z.RELEASE-2022'
    az_fold = 'C:/Users/tonyt/Documents/Research/datasets/solar_azimuth/harv_2022'

    test = StructureDataset(pca_fold, chm_fold, az_fold, 40)

    # test = MaskedDenseVitDataset(pca_fold, 8, eval=True)

    print(test.__getitem__(69).shape)
    #test = pylas.read('/data/shared/src/aalbanese/datasets/lidar/NEON_lidar-point-cloud-line/NEON.D16.WREF.DP1.30003.001.2021-07.basic.20220330T163527Z.PROVISIONAL/NEON_D16_WREF_DP1_L001-1_2021071815_unclassified_point_cloud.las')    
    #print(test)

    #las_fold = "/data/shared/src/aalbanese/datasets/lidar/NEON_lidar-point-cloud-line/NEON.D16.WREF.DP1.30003.001.2021-07.basic.20220330T192134Z.PROVISIONAL"
    # h5_fold = "/data/shared/src/aalbanese/datasets/hs/NEON_refl-surf-dir-ortho-mosaic/NEON.D16.WREF.DP3.30006.001.2021-07.basic.20220330T192306Z.PROVISIONAL"

    # wavelengths = {"red": 654,
    #                 "green": 561, 
    #                 "blue": 482,
    #                 "nir": 865}


    # #test = MixedDataset(h5_fold, las_fold, waves=wavelengths)
    # test = HyperDataset(h5_fold, waves=wavelengths, augment="wavelength")
    # test_item = test.__getitem__(69)
    # print(test_item)


#WIP - COME BACK TO WHEN WORKING ON LIDAR
# class MixedDataset(Dataset):
#     def __init__(self, hyper_folder, las_folder, **kwargs):
#         self.kwargs = kwargs
#         self.h5_location = hyper_folder
#         self.las_location = las_folder
#         self.batch_size = kwargs["batch_size"] if "batch_size" in kwargs else 32
#         self.crop_size = kwargs["crop_size"] if "crop_size" in kwargs else 64
#         h5_files = [file for file in os.listdir(self.h5_location) if ".h5" in file]
#         las_files = [file for file in os.listdir(self.las_location) if ".las" in file]

#         def make_dict(file_list, param_1, param_2):
#             return {(file.split('_')[param_1], file.split('_')[param_2]): file for file in file_list}

#         self.h5_dict = make_dict(h5_files, -3, -2)
#         self.las_dict = make_dict(las_files, -6, -5)

#         self.common_files = list(set(self.h5_dict.keys()) & set(self.las_dict.keys()))
    


#     def process_h5(self, h5_file):
#         waves = self.kwargs['waves']
#         bands, meta, _ = hp.pre_processing(h5_file, waves)
#         return bands, meta

#     def process_lidar(self, lidar_file):
#         lidar = pylas.read(lidar_file)
#         return lidar

#     def make_crops(self):
#         #There is definitely a faster way to do this but it works
#         crops = []
#         while len(crops) < self.batch_size:
#             x_min, y_min = random.randint(0,1000-self.crop_size), random.randint(0,1000-self.crop_size)
#             x_max, y_max = x_min + self.crop_size, y_min + self.crop_size
#             crop_dims = (x_min, y_min, x_max, y_max)
#             x_range = range(x_min, x_max)
#             y_range = range(y_min, y_max)
#             for crop in crops:
#                 crop_x_range = range(crop[0], crop[2])
#                 crop_y_range = range(crop[1], crop[3])
#                 x_int = list(set(x_range) & set(crop_x_range))
#                 y_int = list(set(y_range) & set(crop_y_range))
#                 if len(x_int) * len(y_int):
#                     break
#             else:
#                 crops.append(crop_dims)
#         return crops

#     # Just for debugging
#     def plot_crops(self):
#         crops = self.make_crops()
#         for crop in crops:
#             x_list = [crop[0], crop[2], crop[2], crop[0], crop[0]]
#             y_list = [crop[1], crop[1], crop[3], crop[3], crop[1]]
#             plt.plot(x_list, y_list)
#         plt.show()

#     def make_h5_stack(self, h5, crops):
#         h5 = hp.stack_all(h5, axis=0)

#         h5_samples = [h5[:, crop[0]:crop[2], crop[1]:crop[3]] for crop in crops]
#         # Convert any NAN values to -1
#         h5_tensor_list = []
#         for sample in h5_samples:
#             sample[sample != sample] = -1
#             h5_tensor_list.append(torch.from_numpy(sample))

#         return torch.stack(h5_tensor_list)

#     def make_las_stack(self, las, coords, crops):
#         adj_crops = []
#         coords = [int(coord) for coord in coords]
#         las_points = []
#         x_copy, y_copy = las.x.copy(), las.y.copy()
#         for crop in crops:
#             x_min, x_max = crop[0] + coords[0], crop[2] + coords[0]
#             y_min, y_max = coords[1] - crop[1] + 1000, coords[1] - crop[3] + 1000
            
#             x_mask = np.bitwise_and(x_copy >= x_min, x_copy <=x_max)
#             y_mask = np.bitwise_and(y_copy <= y_min, y_copy >= y_max)
#             mask = np.bitwise_and(x_mask, y_mask)
#             stacked = np.stack((las.x[mask], las.y[mask], las.z[mask]), axis=1)
#             las_points.append(torch.from_numpy(stacked))
        
#         return las_points

    
#     def __len__(self):
#         return len(self.common_files)

#     def __getitem__(self, idx):
#         coords = self.common_files[idx]
#         h5 = self.h5_dict[coords]
#         las = self.las_dict[coords]

#         h5, h5_meta = self.process_h5(os.path.join(self.h5_location, h5))
#         las = self.process_lidar(os.path.join(self.las_location, las))

#         crops = self.make_crops()
        
#         h5_stack = self.make_h5_stack(h5, crops)
#         las_stack = self.make_las_stack(las, coords, crops)
        
#         return {"hs":h5_stack, "las":las_stack}




