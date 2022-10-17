from einops import rearrange
from torch.utils.data import Dataset
import os
import numpy as np
from PIL import Image
import torch
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import torchvision.transforms.functional as ttf
import torchvision.transforms as tt

def fix_labels(arr, forward=True):
    label_keys = [
        (0,-1),
        (1,0),
        (2,1),
        (3,2),
        (4,3),
        (5,4),
        (6,5),
        (8,6),
        (9,7),
        (13,8),
        (14,9),
        (15,10),
        (16,11),
        (36,12)
    ]

    if forward:
        for k_v in label_keys:
            arr[arr == k_v[0]] = k_v[1]
    else:
        for k_v in label_keys.reverse():
            arr[arr==k_v[1]] = k_v[0]
    return arr


class SentinelDataLoader(Dataset):
    def __init__(self,
            base_dir,
            target_dir,
            stats_loc,
            testing=False,
            crop_size = None):
        self.base_dir = base_dir
        self.target_dir = target_dir
        self.stats_loc = stats_loc
        self.testing = testing
        self.crop_size = crop_size
        if crop_size is not None:
            self.crop = tt.RandomCrop(crop_size)
        targets_list = [f.split('_')[-1] for f in os.listdir(target_dir) if '.json' not in f]
        self.all_folders = [f for f in os.listdir(base_dir) if '.json' not in f and f.split('_')[-1] in targets_list]

        self.bands = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12']

        self.calc_stats = True       
        self.scaler = StandardScaler()
        
        
        if os.path.exists(stats_loc):
            self.stats = torch.load(stats_loc)
            self.scaler.scale_ = self.stats['scale']
            self.scaler.mean_ = self.stats['mean']
            self.scaler.var_ = self.stats['var']
            self.calc_stats = False

    def save_stats(self):
        to_save = {'scale': self.scaler.scale_, 'mean': self.scaler.mean_, 'var': self.scaler.var_}
        #with open(self.stats_loc, 'wb') as f:
        torch.save(to_save, self.stats_loc)

        
    
    def __len__(self):
        return len(self.all_folders)
    
    def __getitem__(self, ix):
        to_open = self.all_folders[ix]
        target_folder = os.path.join(self.target_dir, 'ref_agrifieldnet_competition_v1_labels_train_' + to_open.split('_')[-1])
        opened = []
        for b in self.bands:
            im = Image.open(os.path.join(self.base_dir, to_open, b +'.tif'))
            im = np.array(im)
            opened.append(im)
        
        base_img = np.stack(opened)/10000
        base_img = base_img.astype(np.float32)

        to_scale = rearrange(base_img, 'c h w -> (h w) c')
        if self.calc_stats:
            self.scaler.partial_fit(to_scale)
        scaled = self.scaler.transform(to_scale)
        base_img = rearrange(scaled, '(h w) c -> c h w', h=256, w=256)

        field_img = Image.open(os.path.join(target_folder, 'field_ids.tif'))
        field_img = np.array(field_img, dtype=np.int16)

        to_return = {'scenes': torch.from_numpy(base_img),
                'field_ids': torch.from_numpy(field_img)}

        if not self.testing:
            targets = Image.open(os.path.join(target_folder, 'raster_labels.tif'))
            targets = np.array(targets, dtype=np.int64)
            
            to_return['targets'] = torch.from_numpy(targets)

        if self.crop_size is not None:
            target_sum = 0
            while target_sum == 0:
                params = self.crop.get_params(to_return['scenes'], (self.crop_size, self.crop_size))

                if 'targets' in to_return:
                    target_sum = ttf.crop(to_return['targets'], *params).sum()
            for k, v in to_return.items():
                to_return[k] = ttf.crop(v, *params)
        
        if 'targets' in to_return:
            to_return['targets'] = fix_labels(to_return['targets'], forward=True)
        return to_return
                





if __name__ == "__main__":


    BASE_DIR = r'C:\Users\tonyt\Documents\agrifield\ref_agrifieldnet_competition_v1_source'
    TARGET_DIR = r'C:\Users\tonyt\Documents\agrifield\train\ref_agrifieldnet_competition_v1_labels_train'
    STATS_LOC = r'C:\Users\tonyt\Documents\agrifield\stats.npy'

    test = SentinelDataLoader(BASE_DIR, TARGET_DIR, STATS_LOC)
    # for x in test:
    #     print(x['scenes'])
    # #test.save_stats()