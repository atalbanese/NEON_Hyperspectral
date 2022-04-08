from calendar import SATURDAY
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import h5_helper as hp
import inference
import os
import models
import utils
from skimage import exposure

class Validator():
    def __init__(self, **kwargs):
        self.file = kwargs["file"]
        self.img_dir = kwargs["img_dir"]
        self.valid_data = self.open_file()
        self.num_clusters = kwargs["num_clusters"]
        self.valid_files = self.get_valid_files()
        self.cluster_dict = self.make_empty_dict()
        #self._confusion_matrix = None

    def open_file(self):
        data = pd.read_csv(self.file, usecols=['easting', 'northing', 'taxonID', 'ninetyCrownDiameter'])
        data = data.dropna()
        data["pixels"] = (data['ninetyCrownDiameter']**2)//1
        data["file_west_bound"] = data["easting"] - data["easting"] % 1000
        data["file_south_bound"] = data["northing"] - data["northing"] % 1000
        data['x_min'] = (data["easting"] %1000-data["ninetyCrownDiameter"]/2)//1
        data['y_min'] = ((1000-data["northing"] %1000)-data["ninetyCrownDiameter"]/2)//1
        data['x_max'] = data['x_min'] + data["ninetyCrownDiameter"]//1
        data['y_max'] = data['y_min'] + data["ninetyCrownDiameter"]//1
        index_names = data[(data['x_min'] <0) | (data['y_min']<0) | (data['x_max'] >999) | (data['y_max']>999)].index
        data = data.drop(index_names)
        data = data.astype({"file_west_bound": int,
                            "file_south_bound": int,
                            'x_min': int,
                            'x_max': int,
                            'y_min': int,
                            'y_max': int})

        data = data.astype({"file_west_bound": str,
                            "file_south_bound": str})

        data['file_coords'] = data['file_west_bound'] + '_' + data['file_south_bound']

        return data

    def get_valid_files(self):
        if self.img_dir is not None:
            coords = list(self.valid_data['file_coords'].unique())

            all_files = os.listdir(self.img_dir)
            valid_files = {coord:os.path.join(self.img_dir,file) for file in all_files for coord in coords if coord in file}
            return valid_files
        else:
            return None

    def make_empty_dict(self):
        keys = self.valid_data["taxonID"].unique()
        template = {key:[0] * self.num_clusters for key in keys}
        return template


    def validate(self, file_coords, f):
        valid = self.valid_data.loc[self.valid_data["file_coords"] == file_coords]
        if len(valid.index>0):
            if isinstance(f, str):
                clustered = np.load(f)
            elif isinstance(f, np.ndarray):
                clustered = f
            else:
                return False
            for _, row in valid.iterrows():
                select = clustered[row['x_min']:row['x_max'], row['y_min']:row['y_max']]
                taxon = row['taxonID']
                groups, counts = np.unique(select, return_counts=True)
                for j, group in np.ndenumerate(groups):
                    self.cluster_dict[taxon][int(group)] += int(counts[j])

    @property
    def confusion_matrix(self):
        df = pd.DataFrame.from_dict(self.cluster_dict, orient='index', dtype=int)
        df["sum"] = df.sum(axis=1)
        sum_row = pd.DataFrame(df.sum(axis=0)).transpose()
        sum_row = sum_row.rename(index={0:'sum'})
        df = pd.concat([df, sum_row])
        return df

    def kappa(self):
        return None
    

def check_predictions(validator: Validator, model, coords, h5_file, save_dir, **kwargs):
    y = inference.do_inference(model, h5_file, **kwargs)
    np.save(save_dir+coords+".npy", y)
    validator.validate(coords, y)
    return None

def check_all(validator: Validator, model, save_dir, **kwargs):
    for coord, file in validator.valid_files.items():
        check_predictions(validator, model, coord, file, save_dir, **kwargs)
    return None


if __name__ == "__main__":
    NUM_CLUSTERS = 20
    NUM_CHANNELS = 30
    IMG_DIR = "/data/shared/src/aalbanese/datasets/hs/NEON_refl-surf-dir-ortho-mosaic/NEON.D16.WREF.DP3.30006.001.2021-07.basic.20220330T192306Z.PROVISIONAL/"
    OUT_NAME = "test_inference_ckpt_6.npy"
    SAVE_DIR = "/data/shared/src/aalbanese/lidar_hs_unsup_dl_model/inf_tests/"
    # h5_file = "/data/shared/src/aalbanese/datasets/hs/NEON_refl-surf-dir-ortho-mosaic/NEON.D16.WREF.DP3.30006.001.2021-07.basic.20220330T192306Z.PROVISIONAL/NEON_D16_WREF_DP3_580000_5075000_reflectance.h5"
    # coords = "580000_5075000"
    VALID = Validator(file="/data/shared/src/aalbanese/datasets/neon_wref.csv", img_dir=IMG_DIR, num_clusters=NUM_CLUSTERS)
    checkpoint = "ckpts/saved/dense_sim_siam_30_channels_10_epochs/epoch=9.ckpt"

    MODEL = inference.load_ckpt(models.DenseSimSiam, checkpoint, num_channels=NUM_CHANNELS, num_classes = NUM_CLUSTERS)
    check_all(VALID, MODEL, SAVE_DIR, n_components=NUM_CHANNELS)
    print(VALID.confusion_matrix)
    VALID.confusion_matrix.to_csv(os.path.join(SAVE_DIR, "epoch_9_conf_matrix.csv"))
