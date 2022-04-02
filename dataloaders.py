import h5_helper as hp
#import pylas
from torch.utils.data import Dataset
import os
import random
import matplotlib.pyplot as plt
import torch
import numpy as np
#from torch_geometric.data import Data

class HyperDataset(Dataset):
    def __init__(self, hyper_folder, **kwargs):
        self.kwargs = kwargs
        self.h5_location = hyper_folder
        self.batch_size = kwargs["batch_size"] if "batch_size" in kwargs else 32
        self.crop_size = kwargs["crop_size"] if "crop_size" in kwargs else 64
        h5_files = [file for file in os.listdir(self.h5_location) if ".h5" in file]
        
        def make_dict(file_list, param_1, param_2):
            return {(file.split('_')[param_1], file.split('_')[param_2]): file for file in file_list}

        self.h5_dict = make_dict(h5_files, -3, -2)

        self.files = list(self.h5_dict.keys())
    

    def process_h5(self, h5_file):
        waves = self.kwargs['waves']
        bands, meta, _ = hp.pre_processing(h5_file, waves)
        return bands, meta


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

    # Just for debugging
    def plot_crops(self):
        crops = self.make_crops()
        for crop in crops:
            x_list = [crop[0], crop[2], crop[2], crop[0], crop[0]]
            y_list = [crop[1], crop[1], crop[3], crop[3], crop[1]]
            plt.plot(x_list, y_list)
        plt.show()

    def make_h5_stack(self, h5, crops):
        h5 = hp.stack_all(h5, axis=0)

        h5_samples = [h5[:, crop[0]:crop[2], crop[1]:crop[3]] for crop in crops]
        # Convert any NAN values to -1
        h5_tensor_list = []
        for sample in h5_samples:
            sample[sample != sample] = -1
            h5_tensor_list.append(torch.from_numpy(sample))

        return torch.stack(h5_tensor_list)

    
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        coords = self.files[idx]
        h5 = self.h5_dict[coords]
        h5, h5_meta = self.process_h5(os.path.join(self.h5_location, h5))
        
        crops = self.make_crops()
                
        return self.make_h5_stack(h5, crops)
    
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




if __name__ == "__main__":
    #test = pylas.read('/data/shared/src/aalbanese/datasets/lidar/NEON_lidar-point-cloud-line/NEON.D16.WREF.DP1.30003.001.2021-07.basic.20220330T163527Z.PROVISIONAL/NEON_D16_WREF_DP1_L001-1_2021071815_unclassified_point_cloud.las')    
    #print(test)

    las_fold = "/data/shared/src/aalbanese/datasets/lidar/NEON_lidar-point-cloud-line/NEON.D16.WREF.DP1.30003.001.2021-07.basic.20220330T192134Z.PROVISIONAL"
    h5_fold = "/data/shared/src/aalbanese/datasets/hs/NEON_refl-surf-dir-ortho-mosaic/NEON.D16.WREF.DP3.30006.001.2021-07.basic.20220330T192306Z.PROVISIONAL"

    wavelengths = {"blue": {"lower": 452, "upper": 512},
                     "green": {"lower": 533, "upper": 590},
                     "red": {"lower": 636, "upper": 673},
                     "nir": {"lower": 851, "upper": 879}}

    #test = MixedDataset(h5_fold, las_fold, waves=wavelengths)
    test = HyperDataset(h5_fold, waves=wavelengths)
    print(test.__getitem__(0))
