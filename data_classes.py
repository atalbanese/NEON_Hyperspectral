import os
import numpy as np
import torch
import itertools
from typing import Literal

class TreeData:
    def __init__(
        self,
         hyperspectral,
         rgb, 
         rgb_mask,
         hyperspectral_mask,
         hyperspectral_bands,
         chm,
         utm_origin,
         taxa,
         plot_id, 
         site_id,
        ):
        self.hyperspectral = hyperspectral
        self.rgb = rgb
        self.rgb_mask = rgb_mask
        self.hyperspectral_mask = hyperspectral_mask
        self.hyperspectral_bands = hyperspectral_bands
        self.chm = chm
        self.utm_origin = utm_origin
        self.taxa = taxa[()]
        self.plot_id = plot_id[()]
        self.site_id = site_id[()]

    @classmethod
    def from_npz(cls, file_loc):
        with np.load(file_loc) as data:
            new_instance = cls(**data)
        return new_instance

    
    def get_masked_hs(self, out_type ='numpy'):
        if out_type == 'numpy':
            return self.hyperspectral[self.hyperspectral_mask]
        if out_type == 'torch':
            return torch.from_numpy(self.hyperspectral[self.hyperspectral_mask])


    def get_masked_rgb(self, out_type ='numpy'):
        if out_type == 'numpy':
            return self.rgb[self.rgb_mask]
        if out_type == 'torch':
            return torch.from_numpy(self.rgb[self.rgb_mask])


class SiteData:
    def __init__(
        self,
        site_dir: str,
        random_seed: int,
        train: float,
        test: float,
        valid: float
        ):

        self.site_dir = site_dir
        self.all_trees = self.find_all_trees()
        self.all_plots = self.find_all_plots()
        self.all_taxa = self.find_all_taxa()

        self.rng = np.random.default_rng(random_seed)
        self.train_proportion = train
        self.test_proportion = test
        self.valid_proportion = valid

        self.training_data = None
        self.testing_data = None
        self.validation_data = None



        print('here')

    
    def find_all_trees(self):
        all_dirs = [os.scandir(d) for d in os.scandir(self.site_dir) if d.is_dir()]
        return [TreeData.from_npz(f.path) for f in itertools.chain(*all_dirs) if f.name.endswith('.npz')]

    def find_all_plots(self):
        plots_dict = dict()
        for tree in self.all_trees:
            if tree.plot_id in plots_dict:
                plots_dict[tree.plot_id].append(tree)
            else:
                plots_dict[tree.plot_id] = [tree]
        return plots_dict

    def find_all_taxa(self):
        taxa_dict = dict()
        for tree in self.all_trees:
            if tree.taxa in taxa_dict:
                taxa_dict[tree.taxa].append(tree)
            else:
                taxa_dict[tree.taxa] = [tree]
        return taxa_dict
    
    def make_splits(self, split_style: Literal["tree_level", "plot_level"]):
        if split_style == "tree_level":
            self.make_tree_level_splits()
        if split_style == "plot_level":
            self.make_plot_level_splits()


if __name__ == "__main__":
    test = SiteData(r'C:\Users\tonyt\Documents\Research\thesis_final\NIWO', random_seed=42)
