import h5_helper as hp
import pylas
from torch.utils.data import Dataset
import os

class MixedDataset(Dataset):
    def __init__(self, hyper_folder, las_folder, **kwargs):
        self.kwargs = kwargs
        self.h5_location = hyper_folder
        self.las_location = las_folder
        h5_files = [file for file in os.listdir(self.h5_location) if ".h5" in file]
        las_files = [file for file in os.listdir(self.las_location) if ".las" in file]

        def make_dict(file_list, param_1, param_2):
            return {(file.split('_')[param_1], file.split('_')[param_2]): file for file in file_list}

        self.h5_dict = make_dict(h5_files, -3, -2)
        self.las_dict = make_dict(las_files, -6, -5)

        self.common_files = list(set(self.h5_dict.keys()) & set(self.las_dict.keys()))
    


    def process_h5(self, h5_file):
        waves = self.kwargs['waves']
        bands, meta, _ = hp.pre_processing(h5_file, waves)
        return bands, meta

    def process_lidar(self, lidar_file):
        lidar = pylas.read(lidar_file)

        return lidar
    
    def __len__(self):
        return len(self.common_files)

    def __getitem__(self, idx):
        coords = self.common_files[idx]
        h5 = self.h5_dict[coords]
        las = self.las_dict[coords]

        h5, h5_meta = self.process_h5(os.path.join(self.h5_location, h5))
        las = self.process_lidar(os.path.join(self.las_location, las))
        print("here")


        return None




if __name__ == "__main__":
    #test = pylas.read('/data/shared/src/aalbanese/datasets/lidar/NEON_lidar-point-cloud-line/NEON.D16.WREF.DP1.30003.001.2021-07.basic.20220330T163527Z.PROVISIONAL/NEON_D16_WREF_DP1_L001-1_2021071815_unclassified_point_cloud.las')    
    #print(test)

    las_fold = "/data/shared/src/aalbanese/datasets/lidar/NEON_lidar-point-cloud-line/NEON.D16.WREF.DP1.30003.001.2021-07.basic.20220330T192134Z.PROVISIONAL"
    h5_fold = "/data/shared/src/aalbanese/process_lidar/NEON_refl-surf-dir-ortho-mosaic/NEON.D16.WREF.DP3.30006.001.2021-07.basic.20220302T173822Z.PROVISIONAL"

    wavelengths = {"blue": {"lower": 452, "upper": 512},
                     "green": {"lower": 533, "upper": 590},
                     "red": {"lower": 636, "upper": 673},
                     "nir": {"lower": 851, "upper": 879}}

    test = MixedDataset(h5_fold, las_fold, waves=wavelengths)
    test.__getitem__(0)
