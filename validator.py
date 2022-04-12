from matplotlib.widgets import Slider

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
        
        self.num_clusters = kwargs["num_clusters"]
        self.site_name = kwargs["site_name"]
        self.valid_data = self.open_file()
        self.valid_files = self.get_valid_files()
        self.cluster_dict = self.make_empty_dict()
        #self._confusion_matrix = None

    def open_file(self):
        data = pd.read_csv(self.file, usecols=['siteID', 'easting', 'northing', 'taxonID', 'ninetyCrownDiameter'])
        data = data.loc[data['siteID'] == self.site_name]
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
        # data['x_coords'] = [data['x_min'], data['x_max'], data['x_max'], data['x_min'], data['x_min']]
        # data['y_coords'] = [data['y_max'], data['y_max'], data['y_min'], data['y_min'], data['y_max']]

        

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

    def plot_tree(self, coord, file, **kwargs):
     
        fig, ax = plt.subplots()
        rgb = hp.pre_processing(file, wavelength_ranges=utils.get_viz_bands())
        rgb = hp.make_rgb(rgb["bands"])
        rgb = exposure.rescale_intensity(rgb)
        ax.imshow(rgb)
        if 'predictions' in kwargs:
            loc = os.path.join(kwargs['predictions'], coord + '.npy')
            y = np.load(loc)
            im = ax.imshow(y, alpha=.2)
            slider = self._make_slider(fig, im)
        data = self.valid_data.loc[self.valid_data['file_coords'] == coord]
        for ix, row in data.iterrows():
            # x = [row['x_min'], row['x_max'], row['x_max'], row['x_min'], row['x_min']]
            # y = [row['y_max'], row['y_max'], row['y_min'], row['y_min'], row['y_max']]
            x= (row['x_min'] + row['x_max'])//2
            y= (row['y_min'] + row['y_max'])//2
            ax.plot(x, y, marker="o")
            ax.annotate(row['taxonID'], (x, y), textcoords='offset points', xytext= (0, 5), ha='center')
        plt.show()
    
    @staticmethod
    def _make_slider(fig, ax):
        axfreq = plt.axes([0.25, 0.1, 0.65, 0.03])
        plt.subplots_adjust(left=0.25, bottom=0.25)
        freq_slider = Slider(
            ax=axfreq,
            label='Alpha',
            valmin=0.0,
            valmax=1.0,
            valstep=0.2,
            valinit=.2,
        )
        def update(val):
            ax.set_alpha(val)
            fig.canvas.draw_idle()
        
        freq_slider.on_changed(update)
        return freq_slider



        #for ix, row in self.valid_data.iterrows():

    def plot_trees(self, **kwargs):
        for key, value in self.valid_files.items():
            self.plot_tree(key, value, **kwargs)

        


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
    np.save(os.path.join(save_dir,coords+".npy"), y)
    validator.validate(coords, y)
    return None

def check_all(validator: Validator, model, save_dir, **kwargs):
    for coord, file in validator.valid_files.items():
        check_predictions(validator, model, coord, file, save_dir, **kwargs)
    return None

def bulk_validation(ckpts_dir, img_dir, save_dir, valid_file, **kwargs):
    for ckpt in os.listdir(ckpts_dir):
         if ".ckpt" in ckpt:
             model = inference.load_ckpt(models.DenseSimSiam, os.path.join(ckpts_dir, ckpt), **kwargs)
             valid = Validator(file=valid_file, img_dir=img_dir, **kwargs)
             ckpt_name = ckpt.replace(".ckpt", "")
             new_dir = os.path.join(save_dir, ckpt_name)
             os.mkdir(new_dir)
             check_all(valid, model, new_dir, n_components=kwargs['num_channels'])
             valid.confusion_matrix.to_csv(os.path.join(save_dir, ckpt_name + "conf_matrix.csv"))
    return True





if __name__ == "__main__":
    NUM_CLUSTERS = 60
    NUM_CHANNELS = 30
    IMG_DIR = "/data/shared/src/aalbanese/datasets/hs/NEON_refl-surf-dir-ortho-mosaic/NEON.D01.HARV.DP3.30006.001.2019-08.basic.20220407T001553Z.RELEASE-2022"
    OUT_NAME = "test_inference_ckpt_6.npy"
    IMG= os.path.join(IMG_DIR, 'NEON_D01_HARV_DP3_728000_4713000_reflectance.h5')
    SAVE_DIR = "ckpts/saved/harv_40_classes/validation"
    VALID_FILE = "/data/shared/src/aalbanese/datasets/neon-allsites-appidv-latest.csv"
    CKPTS_DIR = "ckpts/saved/harv_40_classes"
    PRED_DIR = 'ckpts/saved/harv_40_classes/validation/harv_densesimsiam_40_classes_epoch=14'
    MODEL = inference.load_ckpt(models.BYOLTransformer, 'ckpts/harv_transformer_60_classes_epoch=0-v2.ckpt')

    test = inference.do_inference(MODEL,IMG ,True, n_components =NUM_CHANNELS)
    rgb = hp.pre_processing(IMG, wavelength_ranges=utils.get_viz_bands())
    rgb = hp.make_rgb(rgb["bands"])
    rgb = exposure.rescale_intensity(rgb, out_range=(0.0,1.0))
    plt.imshow(rgb)
    plt.show()
    plt.imshow(test)
    plt.show()
    print(test)
    # valid = Validator(file=VALID_FILE, img_dir=IMG_DIR, site_name='HARV', num_clusters=NUM_CLUSTERS)
    # valid.plot_trees(predictions = PRED_DIR)

    #bulk_validation(CKPTS_DIR, IMG_DIR, SAVE_DIR, VALID_FILE, site_name='HARV',num_channels=NUM_CHANNELS, num_classes=NUM_CLUSTERS, num_clusters=NUM_CLUSTERS)

    # h5_file = "/data/shared/src/aalbanese/datasets/hs/NEON_refl-surf-dir-ortho-mosaic/NEON.D16.WREF.DP3.30006.001.2021-07.basic.20220330T192306Z.PROVISIONAL/NEON_D16_WREF_DP3_580000_5075000_reflectance.h5"
    # coords = "580000_5075000"
    # VALID = Validator(file="/data/shared/src/aalbanese/datasets/neon_wref.csv", img_dir=IMG_DIR, num_clusters=NUM_CLUSTERS)
    # checkpoint = "ckpts/saved/dense_sim_siam_30_channels_10_epochs/epoch=9.ckpt"

    # MODEL = inference.load_ckpt(models.DenseSimSiam, checkpoint, num_channels=NUM_CHANNELS, num_classes = NUM_CLUSTERS)
    # check_all(VALID, MODEL, SAVE_DIR, n_components=NUM_CHANNELS)
    # print(VALID.confusion_matrix)
    # VALID.confusion_matrix.to_csv(os.path.join(SAVE_DIR, "epoch_9_conf_matrix.csv"))


