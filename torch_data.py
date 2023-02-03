import torch
from torch.utils.data import Dataset
import numpy as np
from einops import rearrange
from typing import Literal

#These are just needed for testing
from splitting import SiteData
from torch.utils.data import DataLoader
#import scipy.ndimage

class TreeDataset(Dataset):
    def __init__(
        self,
        tree_list,
        output_mode: Literal["full", "pixel", "flat_padded"],
        pad_length = 16
    ):
        self.tree_list = tree_list
        self.output_mode = output_mode
        #self.set_output_mode(output_mode)
        self.pad_length = pad_length

        if output_mode == 'pixel':
            print('Pixel mode does not output consistent sequence sizes and cannot be used with a torch DataLoader. Use mode padded_flat instead')

    def fix_2d(self, item):
        return torch.from_numpy(item).float()

    def fix_3d(self, item):
        item = rearrange(item, 'h w c -> c h w')
        return torch.from_numpy(item).float()
    
    def calculate_max_pad(self):
        max_pad = 0
        for tree in self.tree_list:
            tree_sum = tree['hs_mask'].sum()
            max_pad = tree_sum if tree_sum > max_pad else max_pad
        return max_pad
    
    def handle_masking(self, arr, small_mask):
        if arr.shape[0] == small_mask.shape[0]:
            return arr[small_mask]
        # if arr.shape[0] == big_mask.shape[0]:
        #     return arr[big_mask]
        else:
            return arr

    def handle_padding(self, arr):
        pad_diff = self.pad_length - arr.shape[0]
        #0 = pay attention, 1= ignore
        pad_mask = np.ones((self.pad_length,), dtype=np.bool8)
        pad_mask[:arr.shape[0]] = False

        arr = np.pad(arr, ((0, pad_diff), (0,0)))

        return arr, pad_mask


    def get_flat_item(self, item):
        #TODO: there is definitely a way to reconcile this with get_pixel item
        assert 'hs_mask' in item, 'no mask found to select pixels'
        out = dict()
        hs_mask = item['hs_mask']

        for k,v in item.items():
            if isinstance(v, np.ndarray):
                if len(v.shape)>1 and v.dtype != np.bool8:
                    v = self.handle_masking(v, hs_mask)
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


    #Only works with HS items rn, need to go back to splitting/treedata and pad the bigger arrays properly
    def get_pixel_item(self, item):
        assert 'hs_mask' in item, 'no mask found to select pixels'
        out = dict()
        hs_mask = item['hs_mask']
        #scaled_mask = scipy.ndimage.zoom(hs_mask, 10.0)

        for k,v in item.items():
            if isinstance(v, np.ndarray):
                if len(v.shape)>1 and v.dtype != np.bool8:
                    v = self.handle_masking(v, hs_mask)
                    out[k] = torch.from_numpy(v).float()
                else:
                    out[k] = torch.from_numpy(v).float()
            else:
                out[k] = v
        return out

    def get_full_patch(self, item):
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

    def handle_output_mode(self, item):
        if self.output_mode == "full":
            return self.get_full_patch(item)
        if self.output_mode == "pixel":
            return self.get_pixel_item(item)
        if self.output_mode == "flat_padded":
            return self.get_flat_item(item)
        
    def __len__(self):
        return len(self.tree_list)
    
    def __getitem__(self, index):

        item = self.tree_list[index]
        return self.handle_output_mode(item)


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

    test_set = TreeDataset(tree_data, output_mode='flat_padded')
    test_loader = DataLoader(test_set, batch_size=len(test_set), num_workers=1)

    for x in test_loader:
        print(x.keys())

    x = test_set.__getitem__(69)

#     print(x)







