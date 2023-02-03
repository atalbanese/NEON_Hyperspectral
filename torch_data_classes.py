import torch
from torch.utils.data import Dataset
import numpy as np
from einops import rearrange
from data_loading_classes import SiteData

class TreeDataset(Dataset):
    def __init__(
        self,
        tree_list
    ):
        self.tree_list = tree_list

    def fix_2d(self, item):
        return torch.from_numpy(item).float()

    def fix_3d(self, item):
        item = rearrange(item, 'h w c -> c h w')
        return torch.from_numpy(item).float()
    
    def get_flat_item(self):
        pass

    def get_pixel_item(self):
        pass

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

    def set_output_mode(self):
        pass
        
    def __len__(self):
        return len(self.tree_list)
    
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


if __name__ == "__main__":
    test = SiteData(
        site_dir = r'C:\Users\tonyt\Documents\Research\thesis_final\NIWO',
        random_seed=42,
        train = 0.6,
        test= 0.1,
        valid = 0.3)

    #test.all_trees[0].apply_hs_filter([[410,1357],[1400,1800],[1965,2490]])

    test.make_splits('plot_level')
    tree_data = test.get_data('training', ['hs', 'chm', 'rgb', 'origin'], 16, make_key=True)

    test_set = TreeDataset(tree_data)

    x = test_set.__getitem__(69)

    print(x)







