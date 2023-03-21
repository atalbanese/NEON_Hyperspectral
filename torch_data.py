import torch
from torch.utils.data import Dataset
import numpy as np
from einops import rearrange
from transforms import NormalizeHS, BrightnessAugment, Blit, Block
#These are just needed for testing
from splitting import SiteData
from torch.utils.data import DataLoader
#import scipy.ndimage

#TODO: Check the numpy/torch inconsistincies

class BaseTreeDataSet(Dataset):
    def __init__(self,
                tree_list,
                stats,
                augments_list
                ):


        self.tree_list = tree_list
        if len(augments_list) > 0:
            self.transforms = torch.nn.Sequential(*self.build_augments(stats, augments_list))
        else:
            self.transforms = None

    def build_augments(self, stats, augments_list):

        augs = []
        if "brightness" in augments_list:
            augs = augs + [BrightnessAugment(0.5)]
        if "blit" in augments_list:
            augs = augs + [Blit()]
        if "block" in augments_list:
            augs = augs + [Block()]
        if "normalize" in augments_list:
            augs = augs + [NormalizeHS(torch.from_numpy(stats['mean']).float(), torch.from_numpy(stats['std']).float())]
        
        return augs

    
    def handle_transforms(self, arr):
        if self.transforms is not None:
            arr = self.transforms(arr)
        return arr

    def __len__(self):
        return len(self.tree_list)
    
    def __getitem__(self, index):
        pass


class BasicTreeDataSet(BaseTreeDataSet):
    def __init__(self, 
                tree_list,
                stats,
                augments_list,
                inp_key
                ):
        super().__init__(tree_list, stats, augments_list)
        self.inp_key = inp_key
    

    def __getitem__(self, index):
        item = self.tree_list[index]
        out = dict()
        inp = item[self.inp_key]
        target = item['pixel_target']
        pad_mask = item['pad_mask']

        out['input'] = self.handle_transforms(torch.from_numpy(inp).float())
        out['target'] = torch.from_numpy(target).long()
        out['pad_mask'] = torch.from_numpy(pad_mask).bool()

        return out
    



class SyntheticPaddedTreeDataSet(BaseTreeDataSet):
    def __init__(self, tree_list, pad_length, num_synth_trees, stats, augments_list, weights=None):
        super().__init__(tree_list, stats, augments_list)
        self.pad_length = pad_length
        self.num_synth_trees = num_synth_trees
        self.rng = np.random.default_rng()
        self.weights = weights
        self.samp_weights = self.get_sampling_weights()
    
    #TODO: Mess around with rng weights to see if we can make up for unbalanced dataset?

    def __len__(self):
        return self.num_synth_trees
    def get_sampling_weights(self):
        if self.weights is not None:
            samp_weights = np.zeros((len(self.tree_list)), dtype=np.float32)
            for ix, tree in enumerate(self.tree_list):
                samp_weights[ix] = self.weights[int(tree['single_target'].item())]
            return samp_weights/sum(samp_weights)
        else:
            return None

    def assemble_tree_pixels(self, tree_samples):
        tree_pixels = []
        tree_targets = []
        synth_tree_len = 0
        for tree in tree_samples:
            tree_pix = self.handle_masking(tree['hs'], tree['hs_mask'])
            tree_target = self.handle_masking(tree['target_arr'], tree['hs_mask'])
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

        if self.weights is not None:
            tree_samples = self.rng.choice(self.tree_list, self.pad_length, replace=False, p=self.samp_weights)
        else:   
            tree_samples = self.rng.choice(self.tree_list, self.pad_length, replace=False)
        tree_pix, tree_targets = self.assemble_tree_pixels(tree_samples)
        synth_tree = np.concatenate(tree_pix)[:self.pad_length,...]
        synth_tree = self.handle_transforms(synth_tree)
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
                all_data,
                stats,
                augments_list,
                sequence_length,
                inp_key,
                ):
        super().__init__(all_data, stats, augments_list)
        self.all_data = all_data
        self.tree_data = all_data[inp_key]
        self.sequence_length = sequence_length
        self.inp_key = inp_key

        self.num_whole_trees = self.tree_data.shape[0]//sequence_length
        self.num_partial_trees = 1 if self.tree_data.shape[0] % sequence_length != 0 else 0

    def __len__(self):
        return self.num_whole_trees + self.num_partial_trees
    
    def __getitem__(self, index):
        out = dict()
        if index + 1 <= self.num_whole_trees:
            inp = self.tree_data[index*self.sequence_length:index*self.sequence_length+self.sequence_length]
            pad_mask = np.zeros((self.sequence_length,), dtype=np.bool_)
            target = self.all_data['pixel_target'][index*self.sequence_length:index*self.sequence_length+self.sequence_length]

        else:
            inp = self.tree_data[index*self.sequence_length:]
            target = self.all_data['pixel_target'][index*self.sequence_length:]
            pad_dif = self.sequence_length - inp.shape[0]

            inp = np.pad(inp, ((0, pad_dif), (0,0)))
            target = np.pad(target, ((0, pad_dif)))
            pad_mask = np.zeros((self.sequence_length,), dtype=np.bool_)
            if pad_dif > 0:
                pad_mask[-pad_dif:] = True
        
        out['input'] = self.handle_transforms(torch.from_numpy(inp).float())
        out['target'] = torch.from_numpy(target).long()
        out['pad_mask'] = torch.from_numpy(pad_mask).bool()


        return out









