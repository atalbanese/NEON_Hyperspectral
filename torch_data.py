import torch
from torch.utils.data import Dataset
import numpy as np
from einops import rearrange
from typing import Literal
from transforms import NormalizeHS
#These are just needed for testing
from splitting import SiteData
from torch.utils.data import DataLoader
#import scipy.ndimage

#TODO: Check the numpy/torch inconsistincies

class BaseTreeDataSet(Dataset):
    def __init__(self,
                tree_list,
                stats
                ):


        self.tree_list = tree_list
        with np.load(stats) as f:
            self.transforms = torch.nn.Sequential(
                NormalizeHS(torch.from_numpy(f['mean']), torch.from_numpy(f['std'])),
            )

    
    def handle_masking_and_transforms(self, arr, mask):
        if arr.shape[0] == mask.shape[0] and arr.shape[1] == mask.shape[1]:
            
            #mask = torch.from_numpy(mask).bool()
            arr = arr[mask]
            if arr.shape[1] == 372:
                arr = torch.from_numpy(arr).float()
                arr = self.transforms(arr)
                arr = arr.numpy()
        
        return arr


    def __len__(self):
        return len(self.tree_list)
    
    def __getitem__(self, index):
        pass

class PaddedTreeDataSet(BaseTreeDataSet):
    def __init__(self, 
                tree_list,
                pad_length,
                stats
                ):
        super().__init__(tree_list, stats)
        self.pad_length = pad_length
    
    def calculate_max_pad(self):
        max_pad = 0
        for tree in self.tree_list:
            tree_sum = tree['hs_mask'].sum()
            max_pad = tree_sum if tree_sum > max_pad else max_pad
        return max_pad
    
    def handle_padding(self, arr):
        pad_diff = self.pad_length - arr.shape[0]
        #0 = pay attention, 1= ignore
        pad_mask = np.ones((self.pad_length,), dtype=np.bool8)
        pad_mask[:arr.shape[0]] = False

        arr = np.pad(arr, ((0, pad_diff), (0,0)))

        return arr, pad_mask

    def __getitem__(self, index):
        item = self.tree_list[index]
        assert 'hs_mask' in item, 'no mask found to select pixels'
        out = dict()
        hs_mask = item['hs_mask']

        for k,v in item.items():
            if isinstance(v, np.ndarray):
                if len(v.shape)>1 and v.dtype != np.bool8:
                    v = self.handle_masking_and_transforms(v, hs_mask)
                    v, pad_mask = self.handle_padding(v)
                    assert v.shape[0] == self.pad_length, 'incorrect padding occured'
                    out[k] = torch.from_numpy(v).float()
                    out[k+'_pad_mask'] = torch.from_numpy(pad_mask).bool()
                elif v.dtype == np.bool8:
                    pass
                elif k == 'single_target':
                    out[k] = torch.from_numpy(v).long()
                else:
                    out[k] = torch.from_numpy(v).float()
            else:
                out[k] = v
        return out

class SyntheticPaddedTreeDataSet(BaseTreeDataSet):
    def __init__(self, tree_list, pad_length, num_synth_trees, num_features, stats):
        super().__init__(tree_list, stats)
        self.pad_length = pad_length
        self.num_synth_trees = num_synth_trees
        self.num_features = num_features
        self.rng = np.random.default_rng(42)
    
    #TODO: Mess around with rng weights to see if we can make up for unbalanced dataset?

    def __len__(self):
        return self.num_synth_trees
    
    def assemble_tree_pixels(self, tree_samples):
        tree_pixels = []
        tree_targets = []
        synth_tree_len = 0
        for tree in tree_samples:
            tree_pix = self.handle_masking_and_transforms(tree['hs'], tree['hs_mask'])
            tree_target = self.handle_masking_and_transforms(tree['target_arr'], tree['hs_mask'])
            num_tree_pixels = tree_pix.shape[0]            
            #Figure out if any pixels will be cropped and adjust the target multiplier accordingly
            remaining_pixels = self.pad_length - (synth_tree_len+num_tree_pixels)
            target_mult = num_tree_pixels + remaining_pixels if remaining_pixels < 0 else num_tree_pixels

            tree_pixels.append(tree_pix)
            tree_targets.append(tree_target* target_mult)

            synth_tree_len += num_tree_pixels
            if synth_tree_len >= self.pad_length:
                break

        return tree_pixels, tree_targets


    def make_synthetic_tree(self):
        tree_samples = self.rng.choice(self.tree_list, self.pad_length, replace=False)

        tree_pix, tree_targets = self.assemble_tree_pixels(tree_samples)
        synth_tree = np.concatenate(tree_pix)[:self.pad_length,...]
        synth_target = np.sum(tree_targets, axis=0)/self.pad_length

        #Done for consistency with other methods but we don't really need it
        pad_mask = torch.zeros((self.pad_length,), dtype=torch.bool)

        out = {'hs': torch.from_numpy(synth_tree).float(),
               'target_arr': torch.from_numpy(synth_target).float(),
               'hs_pad_mask': pad_mask}

        return out

    def __getitem__(self, index):
        return self.make_synthetic_tree()


class PixelTreeDataSet(BaseTreeDataSet):
    def __init__(self, 
                tree_list,
                stats
                ):
        super().__init__(tree_list, stats)
    
    def __getitem__(self, index):
        item = self.tree_list[index]
        assert 'hs_mask' in item, 'no mask found to select pixels'
        out = dict()
        hs_mask = item['hs_mask']
        #scaled_mask = scipy.ndimage.zoom(hs_mask, 10.0)

        for k,v in item.items():
            if isinstance(v, np.ndarray):
                if len(v.shape)>1 and v.dtype != np.bool8:
                    v = self.handle_masking_and_transforms(v, hs_mask)
                    out[k] = torch.from_numpy(v).float()
                else:
                    out[k] = torch.from_numpy(v).float()
            else:
                out[k] = v
        return out

class FullTreeDataSet(BaseTreeDataSet):
    def __init__(self, 
                tree_list,
                stats
                ):
        super().__init__(tree_list, stats)

    def fix_3d(self, item):
        item = rearrange(item, 'h w c -> c h w')
        return torch.from_numpy(item).float()
    
    def fix_2d(self, item):
        return torch.from_numpy(item).float()
    
    def __getitem__(self, index):
        item = self.tree_list[index]
        out = dict()
        for k, v in item.items():
            if isinstance(v, np.ndarray):
                if len(v.shape) == 2:
                    out[k] = self.fix_2d(v)
                if len(v.shape) == 3:
                    out[k] = self.fix_3d(v)
                else:
                    out[k] = torch.from_numpy(v).float()
            else:
                out[k] = v
        
        return out

def calc_stats(full_batch_dl, save_loc):
    for x in full_batch_dl:
        pass
    masked = x['hs'][~x['hs_pad_mask']]
    hs_mean = masked.mean(dim=0)
    hs_std = masked.std(dim=0)

    hs_mean = hs_mean.numpy()
    hs_std = hs_std.numpy()

    np.savez(save_loc, mean=hs_mean, std=hs_std)



if __name__ == "__main__":
    test = SiteData(
        site_dir = r'C:\Users\tonyt\Documents\Research\thesis_final\NIWO',
        random_seed=42,
        train = 0.6,
        test= 0.1,
        valid = 0.3)

    #test.all_trees[0].apply_hs_filter([[410,1357],[1400,1800],[1965,2490]])

    test.make_splits('plot_level')
    tree_data = test.get_data('training', ['hs', 'origin'], 16, make_key=True)

    test_set = SyntheticPaddedTreeDataSet(tree_list = tree_data, pad_length=16, num_synth_trees=5000, num_features=372, stats = 'stats/niwo_stats.npz')

    #test_set = PaddedTreeDataSet(tree_data, 16, 'stats/niwo_stats.npz')
    test_loader = DataLoader(test_set, batch_size=len(test_set), num_workers=1)

    #calc_stats(test_loader, 'stats/niwo_stats.npz')

    for x in test_loader:
        print(x.keys())

    x = test_set.__getitem__(69)

    # print(x)






