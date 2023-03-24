import torch
from torch.utils.data import Dataset
import numpy as np
from einops import rearrange
import os
from transforms import NormalizeHS, BrightnessAugment, Blit, Block

class PreTrainingData(Dataset):
    def __init__(self,
                data_dir: str,
                sequence_length: int,
                sitename: str,
                augments_list: list,
                file_dim: int
                ):

        assert (file_dim*file_dim/sequence_length) % 1 == 0, "files must be able to divide cleanly into sequences"
        assert (np.sqrt(sequence_length)) % 1 == 0, "sequence length must be able to make a nice square"

        self.file_dim = file_dim
        self.sequence_sqrt = int(np.sqrt(sequence_length))
        
        self.data_files = [f for f in os.scandir(data_dir) if f.name.endswith(".npy")]
        self.num_files = len(self.data_files)
        self.sitename = sitename
        self.sequence_length = sequence_length

        self.num_entries = (len(self.data_files) * self.file_dim * self.file_dim)//sequence_length
        self.entries_in_file = self.file_dim*self.file_dim//sequence_length
        
        self.columns = int(np.sqrt((self.file_dim*self.file_dim)/self.sequence_length))
        self.rows = int(np.sqrt((self.file_dim*self.file_dim)/self.sequence_length))
        self.open_files = dict()
        self.all_data, self.stats = self.gather_data()


        if len(augments_list) > 0:
            self.transforms = torch.nn.Sequential(*self.build_augments(self.stats, augments_list))
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
    
    def gather_data(self):
        all_data = []
        for ix in range(self.num_entries):
            chunk = self.grab_chunk(ix)
            missing_data_mask = self.get_mask(chunk)
            if missing_data_mask.sum()>self.sequence_length//2:
                all_data.append(chunk)
        stacked_data = np.stack(all_data)
        mean = np.nanmean(stacked_data, axis=(0,1,2))
        std = np.nanstd(stacked_data, axis=(0,1,2))
        #Hack to prevent divide by 0 in case there is a channel with 0 standard deviation - it occured once I swear it!
        std[std == 0] = 0.00001
        return all_data, {'mean': mean, 'std': std}

    def get_mask(self, chunk):
        missing_data_mask = chunk == chunk
        return np.logical_and.reduce(missing_data_mask, axis=(2))


    def grab_chunk(self, index):
        file_index = index//self.entries_in_file
        sequence_index = index % self.entries_in_file
        open_file = self.data_files[file_index]

        min_y, max_y, min_x, max_x = self.get_bounds(sequence_index)

        if open_file not in self.open_files:
            opened = np.load(open_file, mmap_mode='r')
            self.open_files[open_file] = opened
        
        grab = self.open_files[open_file][min_y:max_y,min_x:max_x,...]

        return grab


    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, index):
        out = dict()

        grab = self.all_data[index]

        mask = self.get_mask(grab)
        pad_dif = np.count_nonzero(~mask)
        grab = np.pad(grab[mask], ((0, pad_dif), (0,0)))

        pad_mask = np.zeros((self.sequence_length,), dtype=np.bool_)
        if pad_dif > 0:
            pad_mask[-pad_dif:] = True

        pad_mask = torch.from_numpy(pad_mask).bool()
        grab = torch.from_numpy(grab).float()
        if self.transforms is not None:
            grab = self.transforms(grab)
        out['input'] = grab
        out['pad_mask'] = pad_mask
        return out


    def get_bounds(self, sequence_index):
        row_start = (sequence_index//self.columns) * self.sequence_sqrt
        col_start = (sequence_index%self.rows) * self.sequence_sqrt
        row_end = row_start + self.sequence_sqrt
        col_end = col_start + self.sequence_sqrt


        return row_start, row_end, col_start, col_end
        

if __name__ == "__main__":
    test = PreTrainingData(
        data_dir="/home/tony/thesis/lidar_hs_unsup_dl_model/final_data/STEI/PCA",
        sequence_length=16,
        sitename="STEI",
        augments_list=["normalize"],
        file_dim=1000,
    )

    test.__getitem__(250)
    test.__getitem__(251)
    test.__getitem__(900000)