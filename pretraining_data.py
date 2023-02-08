import torch
from torch.utils.data import Dataset
import numpy as np
from einops import rearrange
import os
from transforms import NormalizeHS, BrightnessAugment, Blit, Block
import h5py as hp

class PreTrainingData(Dataset):
    def __init__(self,
                data_dir: str,
                sequence_length: int,
                hs_filters: list,
                sitename: str,
                augments_list: list,
                stats: str,
                file_dim: int
                ):

        assert (file_dim*file_dim/sequence_length) % 1 == 0, "files must be able to divide cleanly into sequences"
        assert (np.sqrt(sequence_length)) % 1 == 0, "sequence length must be able to make a nice square"

        self.file_dim = file_dim
        self.sequence_sqrt = int(np.sqrt(sequence_length))
        
        self.data_files = [hp.File(f.path, 'r') for f in os.scandir(data_dir) if f.name.endswith(".h5")]
        self.num_files = len(self.data_files)
        self.hs_filters = hs_filters
        self.sitename = sitename
        self.sequence_length = sequence_length

        self.num_entries = (len(self.data_files) * self.file_dim * self.file_dim)//sequence_length
        self.entries_in_file = self.file_dim*self.file_dim//sequence_length

        if len(augments_list) > 0:
            self.transforms = torch.nn.Sequential(*self.build_augments(stats, augments_list))
        else:
            self.transforms = None

        self.columns = int(np.sqrt((self.file_dim*self.file_dim)/self.sequence_length))
        self.rows = int(np.sqrt((self.file_dim*self.file_dim)/self.sequence_length))

        self.hs_indexes = self.get_hs_filter(self.data_files[0][self.sitename]["Reflectance"]["Metadata"]['Spectral_Data']['Wavelength'][:])

    def build_augments(self, stats_loc, augments_list):

        augs = []
        if "brightness" in augments_list:
            augs = augs + [BrightnessAugment(0.5)]
        if "blit" in augments_list:
            augs = augs + [Blit()]
        if "block" in augments_list:
            augs = augs + [Block()]
        if "normalize" in augments_list:
            with np.load(stats_loc) as f:
                augs = augs + [NormalizeHS(torch.from_numpy(f['mean']), torch.from_numpy(f['std']))]
        
        return augs

    def __len__(self):
        return self.num_entries

    def __getitem__(self, index):
        file_index = index//self.entries_in_file
        sequence_index = index % self.entries_in_file

        hs_file = self.data_files[file_index]

        min_y, max_y, min_x, max_x = self.get_bounds(sequence_index)

        #with hp.File(to_open.path, 'r') as hs_file:
        #bands = hs_file[self.sitename]["Reflectance"]["Metadata"]['Spectral_Data']['Wavelength'][:]
        #hs_indexes = self.get_hs_filter(bands)
        hs_grab = hs_file[self.sitename]["Reflectance"]["Reflectance_Data"][min_y:max_y,min_x:max_x, self.hs_indexes]/10000
        hs_grab[hs_grab<0] = 0
        hs_grab[hs_grab>1] = 1

        hs_grab = torch.from_numpy(hs_grab).float()
        if self.transforms is not None:
            hs_grab = self.transforms(hs_grab)
        return rearrange(hs_grab, 'h w c -> (h w) c')

    def get_bounds(self, sequence_index):
        row_start = (sequence_index//self.columns) * self.sequence_sqrt
        col_start = (sequence_index%self.rows) * self.sequence_sqrt
        row_end = row_start + self.sequence_sqrt
        col_end = col_start + self.sequence_sqrt


        return row_start, row_end, col_start, col_end
        

    def get_hs_filter(self, bands):
        # hs_filter should be a list of [min, max]
        mask_list = [(bands>=lmin) & (bands<=lmax) for lmin, lmax in self.hs_filters]
        band_mask = np.logical_or.reduce(mask_list)
        idxs = np.where(band_mask)[0]
        return idxs

if __name__ == "__main__":
    test = PreTrainingData(
        data_dir="/home/tony/thesis/data/NIWO_unlabeled",
        sequence_length=16,
        hs_filters=[[410,1357],[1400,1800],[1965,2485]],
        sitename="NIWO",
        augments_list=["normalize"],
        file_dim=1000,
        stats="/home/tony/thesis/data/stats/niwo_stats.npz"
    )

    test.__getitem__(250)
    test.__getitem__(251)
    test.__getitem__(900000)