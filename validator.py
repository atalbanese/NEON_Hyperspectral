from select import select
from imageio import save
from matplotlib.widgets import Slider
import geopandas as gpd
from rasterio.plot import show
import rasterio as rs
from rasterio.transform import from_origin
import rasterio.features as rf
from rasterio.crs import CRS
import torchvision.transforms.functional as TF
import torch
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import h5_helper as hp
import os
import utils
from skimage import exposure
from scipy.stats import linregress
from skimage.segmentation import mark_boundaries
from einops import rearrange
from sklearn.decomposition import PCA, IncrementalPCA
import torchvision.transforms as tt
import sklearn.model_selection as ms
import sklearn.utils.class_weight as cw
import math


#TODO: Clean this mess up
#Switch plots to plotly? Good for overlays
class Validator():
    def __init__(self, struct=False, rescale=False, train_split=0.6, valid_split=0.2, test_split=0.2, use_sp=True, scholl_filter=False, scholl_output=False, filter_species=None,  **kwargs):
        self.rescale=rescale
        self.file = kwargs["file"]
        self.pca_dir = kwargs["pca_dir"]
        self.ica_dir = kwargs['ica_dir']
        self.raw_bands_dir = kwargs['raw_bands']
        self.shadow_dir = kwargs['shadow']
        self.indexes_dir = kwargs['indexes']
        self.sp_dir = kwargs['superpixel']
        self.struct = struct
        self.curated = kwargs['curated']
        self.site_prefix = kwargs['prefix']

        
        self.num_classes = kwargs["num_classes"]
        self.site_name = kwargs["site_name"]
        self.plot_file = kwargs['plot_file']
        self.orig_dir = kwargs['orig']
        self.chm_mean = kwargs['chm_mean']
        self.chm_std = kwargs['chm_std']
        self.use_sp = use_sp
        self.scholl_filter = scholl_filter
        self.scholl_output = scholl_output
        self.filter_species = filter_species

        self.valid_data, self.data_gdf = self.get_plot_data()

        self.valid_files = self.get_valid_files()

        self.valid_dict = self.make_valid_dict()
        
        

        self.orig_files = [os.path.join(kwargs['orig'], file) for file in os.listdir(kwargs['orig']) if ".h5" in file]
        self.pca_files = [os.path.join(kwargs['pca_dir'], file) for file in os.listdir(kwargs['pca_dir']) if ".npy" in file]
        
        def make_dict(file_list, param_1, param_2):
            return {f"{file.split('_')[param_1]}_{file.split('_')[param_2]}": file for file in file_list}

        self.orig_dict = make_dict(self.orig_files, -3, -2)
        self.pca_dict = make_dict(self.pca_files, -4, -3)

        self.indexes_files = [os.path.join(kwargs['indexes'], file) for file in os.listdir(kwargs['indexes']) if ".npy" in file]


        self.chm_files = [os.path.join(kwargs['chm'], file) for file in os.listdir(kwargs['chm']) if ".tif" in file]
        self.azm_files = [os.path.join(kwargs['azm'], file) for file in os.listdir(kwargs['azm']) if ".npy" in file]

        self.sp_files = [os.path.join(self.sp_dir, file) for file in os.listdir(self.sp_dir) if ".npy" in file]
        self.sp_dict = make_dict(self.sp_files, -4, -3)

        self.chm_dict = make_dict(self.chm_files, -3, -2)
        self.azm_dict = make_dict(self.azm_files, -4, -3)

        self.ica_files = [os.path.join(self.ica_dir, file) for file in os.listdir(self.ica_dir) if ".npy" in file]
        self.extra_files = [os.path.join(self.raw_bands_dir, file) for file in os.listdir(self.raw_bands_dir) if ".npy" in file]
        self.shadow_files = [os.path.join(self.shadow_dir, file) for file in os.listdir(self.shadow_dir) if ".npy" in file]

        self.ica_files_dict = make_dict(self.ica_files, -5, -4)
        self.extra_files_dict = make_dict(self.extra_files, -4, -3)
        self.shadow_dict = make_dict(self.shadow_files, -4, -3)
        self.indexes_dict = make_dict(self.indexes_files, -4, -3)

        self.cluster_groups = set()

        self.last_cluster = {}
        self.rng = np.random.default_rng(42)
        
        if use_sp:
            self.data_gdf = self.pick_superpixels()
        self.taxa = {key: ix for ix, key in enumerate(self.data_gdf['taxonID'].unique())}

        self.train, self.valid, self.test = self.get_splits(train_split, valid_split, test_split)
        self.class_weights = cw.compute_class_weight(class_weight='balanced', classes=self.data_gdf['taxonID'].unique(), y=self.train['taxonID'])
        print('here')

    @staticmethod
    def get_crop(arr, ix):
            locs = arr != ix
            falsecols = np.all(locs, axis=0)
            falserows = np.all(locs, axis=1)

            firstcol = falsecols.argmin()
            firstrow = falserows.argmin()

            lastcol = len(falsecols) - falsecols[::-1].argmin()
            lastrow = len(falserows) - falserows[::-1].argmin()

            return firstrow,lastrow,firstcol,lastcol

    @staticmethod
    def get_pad(arr, pad_size):
            row_len = arr.shape[0]
            col_len = arr.shape[1]

            row_pad = (pad_size - row_len) // 2
            col_pad = (pad_size - col_len) // 2
            
            add_row = (row_pad*2 + row_len) != pad_size
            add_col = (col_pad*2 + col_len) != pad_size

            return [(row_pad, row_pad+add_row), (col_pad, col_pad+add_col)]


    def render_valid_data(self, save_dir, split, out_size=3, target_size=3):
        if split == 'train':
            data = self.train
        if split == 'valid':
            data = self.valid
        if split == 'test':
            data = self.test

        loaded_key = None
        for ix, row in data.iterrows():
            key = row['file_coords']
            if key != loaded_key:
                pca = np.load(self.pca_dict[key]).astype(np.float32)
                
     
                #Azimuth
                azm = np.load(self.azm_dict[key]).astype(np.float32)
                azm = (azm-180)/180
                azm[azm != azm] = 0

                #CHM
                chm_open = rs.open(self.chm_dict[key])
                chm = chm_open.read().astype(np.float32)
                chm[chm==-9999] = np.nan
                chm = chm.squeeze(axis=0)
                chm[chm != chm] = 0

                #ICA
                ica = np.load(self.ica_files_dict[key]).astype(np.float32)

                #Shadow Index
                shadow = np.load(self.shadow_dict[key]).astype(np.float32)

                #Raw bands
                extra = np.load(self.extra_files_dict[key]).astype(np.float32)

                indexes = np.load(self.indexes_dict[key]).astype(np.float32)

                indexes = rearrange(indexes, 'c h w -> h w c')

                sp = np.load(self.sp_dict[key])
                loaded_key = key

            taxa = row['taxonID']
            
            height = float(row['height'])


            rad = out_size//2 
            if self.use_sp:
                super_pix_num = row['sp']

                sp_mask = sp == super_pix_num
                masked_chm = chm * sp_mask
                max_height = masked_chm.max()
                height_pix = masked_chm == max_height
                h_col = np.all(~height_pix, axis=0).argmin()
                h_row = np.all(~height_pix, axis=1).argmin()


                #NEED TO CHECK FOR OVERLAP SOMEHOW
                bounds = (h_row - rad, h_row+rad+1, h_col-rad, h_col+rad+1)
            elif self.scholl_output:
                t_col = int(row['tree_x'])
                t_row = int(row['tree_y'])
                #LOOKUP HOW I USED TO DO THIS
                bounds = (t_row - rad, t_row+rad+1, t_col-rad, t_col+rad+1)
                


            #Just grabbing 3x3 crops right now
            #y_pad = out_size - (bounds[1]-bounds[0])
            #x_pad = out_size - (bounds[3]-bounds[2])

            chm_crop = chm[bounds[0]:bounds[1], bounds[2]:bounds[3]]
            #chm_crop = np.pad(chm_crop, ((0, y_pad), (0, x_pad)), mode='constant', constant_values=-9999)
            

            pca_crop = pca[bounds[0]:bounds[1], bounds[2]:bounds[3],:]
            #pca_crop = np.pad(pca_crop, ((0, y_pad), (0, x_pad), (0, 0)), mode='constant', constant_values=-9999)

            #if pca_crop.shape == (4, 4, 10):
            ica_crop = ica[bounds[0]:bounds[1], bounds[2]:bounds[3],:]
            #ica_crop = np.pad(ica_crop, ((0, y_pad), (0, x_pad), (0, 0)), mode='constant', constant_values=-9999)
            
            azm_crop = azm[bounds[0]:bounds[1], bounds[2]:bounds[3]]
            #azm_crop = np.pad(azm_crop, ((0, y_pad), (0, x_pad)), mode='constant', constant_values=-9999)
            shadow_crop = shadow[bounds[0]:bounds[1], bounds[2]:bounds[3]]
            #shadow_crop = np.pad(shadow_crop, ((0, y_pad), (0, x_pad)), mode='constant', constant_values=-9999)
            extra_crop = extra[bounds[0]:bounds[1], bounds[2]:bounds[3]]
            #extra_crop = np.pad(extra_crop, ((0, y_pad), (0, x_pad), (0, 0)), mode='constant', constant_values=-9999)

            index_crop = indexes[bounds[0]:bounds[1], bounds[2]:bounds[3]]


            mask = pca_crop != pca_crop

            if pca_crop.shape == (out_size, out_size, 10):
                pca_crop = torch.tensor(pca_crop)
                pca_crop = rearrange(pca_crop, 'h w c -> c h w')
                ica_crop = torch.tensor(ica_crop)
                ica_crop = rearrange(ica_crop, 'h w c -> c h w')
                shadow_crop = torch.tensor(shadow_crop)
                extra_crop = torch.tensor(extra_crop)
                extra_crop = rearrange(extra_crop, 'h w c -> c h w')
                chm_crop = torch.tensor(chm_crop)
                azm_crop = torch.tensor(azm_crop)
                mask = torch.tensor(mask)
                mask = rearrange(mask, 'h w c -> c h w')
                index_crop = torch.tensor(index_crop)
                index_crop = rearrange(index_crop, 'h w c -> c h w')

                label = torch.zeros((len(self.taxa.keys()),target_size, target_size), dtype=torch.float32).clone()
                label[self.taxa[taxa]] = 1.0

                to_save = {
                    'pca': pca_crop,
                    'ica': ica_crop,
                    'shadow': shadow_crop,
                    'raw_bands': extra_crop,
                    'chm': chm_crop,
                    'azm': azm_crop,
                    'mask': mask,
                    'target': label,
                    'height': height,
                    'indexes': index_crop
                }

                f_name = f'{key}_{row["taxonID"]}_{row["individualID"]}.pt'
                with open(os.path.join(save_dir, f_name), 'wb') as f:
                    torch.save(to_save, f)
            
        return None

    def render_valid_patch(self, save_dir, split, out_size=20, target_size=3, multi_crop=1, num_channels=16, key_label='pca'):
        if split == 'train':
            data = self.train
        if split == 'valid':
            data = self.valid
        if split == 'test':
            data = self.test

        loaded_key = None
        for ix, row in data.iterrows():
            key = row['file_coords']
            if key != loaded_key:
                pca = np.load(self.pca_dict[key]).astype(np.float32)

            taxa = row['taxonID']
            
            height = float(row['height'])


            rad = 1

            t_col = int(row['tree_x'])
            t_row = int(row['tree_y'])

            target_bounds = (t_row - rad, t_row+rad+1, t_col-rad, t_col+rad+1)

            for ix in range(multi_crop):
                neg_x_pad = self.rng.integers(0, out_size-target_size)
                pos_x_pad = (out_size-target_size) - neg_x_pad
                neg_y_pad = self.rng.integers(0, out_size-target_size)
                pos_y_pad = (out_size-target_size) - neg_y_pad

                x_min = target_bounds[2] - neg_x_pad
                x_max = target_bounds[3] + pos_x_pad

                y_min = target_bounds[0] - neg_y_pad
                y_max = target_bounds[1] + pos_y_pad

                pca_crop = pca[y_min:y_max, x_min:x_max,:]

                crop_coords = (neg_y_pad, neg_x_pad, 3, 3)
                #pca_crop = np.pad(pca_crop, ((0, y_pad), (0, x_pad), (0, 0)), mode='constant', constant_values=-9999)

                
                

                if pca_crop.shape == (out_size, out_size, num_channels):
                    pca_crop = torch.tensor(pca_crop)
                    pca_crop = rearrange(pca_crop, 'h w c -> c h w')
                    mask = pca_crop != pca_crop
                    pca_crop[mask] = 0
                    

                    label = torch.zeros((len(self.taxa.keys()),target_size, target_size), dtype=torch.float32).clone()
                    label[self.taxa[taxa]] = 1.0

                    to_save = {
                        key_label: pca_crop,
                        'mask': mask,
                        'target': label,
                        'height': height,
                        'crop_coords': crop_coords
                    }

                    f_name = f'{key}_{row["taxonID"]}_{row["individualID"]}_multi_crop_{ix}.pt'
                    with open(os.path.join(save_dir, f_name), 'wb') as f:
                        torch.save(to_save, f)
            
        return None


    def make_valid_dict(self):
        species = self.valid_data['taxonID'].unique()
        valid_dict = {}
        for specie in species:
            valid_dict[specie] = {'expected':0,
                                 'found': {i:0 for i in range(self.num_classes)}}
        return valid_dict

    @staticmethod    
    def _gdf_validate_taxon(df, transform, img, vd):
        if len(df) >0:
            taxa = df.iloc[0]['taxonID']
            tree_outlines = rf.geometry_mask(df.crowns, (1000,1000), transform=transform, invert=True)
            trees = np.ma.masked_where(tree_outlines == False, tree_outlines)
            selected = img[trees]
            vd.valid_dict[taxa]['expected'] += trees.sum()
            id, counts = np.unique(selected, return_counts=True)
            for i, count in zip(id, counts):
                vd.valid_dict[taxa]['found'][i] += count
        else:
            print('here')
            print(df)

        return df

    
    def validate_from_gdf(self, file_key, img):
        west, south = file_key.split('_')
        west, south = int(west), int(south)
        cur_gdf = self.data_gdf.loc[(self.data_gdf.easting > west) & (self.data_gdf.easting < west+1000) & (self.data_gdf.northing > south) & (self.data_gdf.northing < south+1000)]
        img_loc = self.valid_files[file_key]
        transform = from_origin(west, south+1000, 1, 1)


        cur_gdf.groupby('taxonID').apply(self._gdf_validate_taxon, transform, img, self) # original, chm)

        

    def get_plot_data(self):
       
        curated = pd.read_csv(self.curated)
        curated = curated.rename(columns={'adjEasting': 'easting',
                                          'adjNorthing': 'northing'})
        data_with_coords = curated.loc[(curated.easting == curated.easting) & (curated.northing==curated.northing) & (curated.maxCrownDiameter == curated.maxCrownDiameter)]
            
        data_gdf = gpd.GeoDataFrame(data_with_coords, geometry=gpd.points_from_xy(data_with_coords.easting, data_with_coords.northing), crs='EPSG:32618')
        data_gdf = data_gdf.loc[data_gdf['taxonID'] != self.filter_species]
        data_gdf['crowns'] = data_gdf.geometry.buffer(data_gdf['maxCrownDiameter']/2)
        data_gdf = data_gdf.loc[data_gdf.crowns.area > 2]

        #TODO: Test with and without this

        if self.scholl_filter:
            to_remove = set()
            for ix, row in data_gdf.iterrows():
                working_copy = data_gdf.loc[data_gdf.index != ix]
                coverage = working_copy.crowns.contains(row.crowns)
                cover_gdf = working_copy.loc[coverage]
                if (cover_gdf['height']>row['height']).sum() > 0:
                    to_remove.add(ix)
                intersect = working_copy.crowns.intersects(row.crowns)
                inter_gdf = working_copy.loc[intersect]
                if (inter_gdf['height']>row['height']).sum() > 0:
                    to_remove.add(ix)

            data_gdf =  data_gdf.drop(list(to_remove))

        data_gdf["file_west_bound"] = data_gdf["easting"] - data_gdf["easting"] % 1000
        data_gdf["file_south_bound"] = data_gdf["northing"] - data_gdf["northing"] % 1000
        data_gdf = data_gdf.astype({"file_west_bound": int,
                            "file_south_bound": int})

        data_gdf['tree_x'] = data_gdf.easting - data_gdf.file_west_bound
        data_gdf['tree_y'] = 1000 - (data_gdf.northing - data_gdf.file_south_bound)

        data_gdf = data_gdf.astype({"file_west_bound": str,
                            "file_south_bound": str})
                            
        data_gdf['file_coords'] = data_gdf['file_west_bound'] + '_' + data_gdf['file_south_bound']

               
        plots = pd.read_csv(self.plot_file, usecols=['plotID', 'siteID', 'subtype', 'easting', 'northing', 'plotSize'])
        plots = plots.loc[plots['siteID'] == self.site_name]
        plots = plots.loc[plots['subtype'] == 'basePlot']

        data = data_gdf.merge(plots, how='left', on='plotID')

        data = data.rename(columns={
                                    'easting_x': 'easting_tree',
                                    'northing_x': 'northing_tree',
                                    'easting_y': 'easting_plot',
                                    'northing_y': 'northing_plot'
        })

        data["file_west_bound"] = data["easting_plot"] - data["easting_plot"] % 1000
        data["file_south_bound"] = data["northing_plot"] - data["northing_plot"] % 1000

        data = data.loc[data['file_west_bound'] == data['file_west_bound']]

        data = data.astype({"file_west_bound": int,
                            "file_south_bound": int})

        data['x_min'] = (data['easting_plot']//1 - data['file_west_bound']) - (data['plotSize']**(1/2)/2)
        data['x_max'] = data['x_min'] + data['plotSize']**(1/2)

        data['y_min'] = 1000- (data['northing_plot']//1 - data['file_south_bound']) - (data['plotSize']**(1/2)/2)
        data['y_max'] = data['y_min'] + data['plotSize']**(1/2)

        data['tree_x'] = data['easting_tree']//1 - data['file_west_bound']
        data['tree_y'] = 1000 - (data['northing_tree']//1 - data['file_south_bound'])

        data = data.astype({"file_west_bound": str,
                            "file_south_bound": str,
                            'x_min':int,
                            'y_min':int,
                            'x_max':int,
                            'y_max': int})
        
        index_names = data[(data['x_min'] <0) | (data['y_min']<0) | (data['x_max'] >999) | (data['y_max']>999)].index
        data = data.drop(index_names)

        data['file_coords'] = data['file_west_bound'] + '_' + data['file_south_bound']
        data = pd.DataFrame(data)

        return data, data_gdf



    def get_valid_files(self):
        if self.pca_dir is not None:
            coords = list(self.valid_data['file_coords'].unique())

            all_files = os.listdir(self.pca_dir)
            valid_files = {coord:os.path.join(self.pca_dir,file) for file in all_files for coord in coords if coord in file}
            return valid_files
        else:
            return None

    def get_splits(self, train_prop, valid_prop, test_prop):
        train, t_v = ms.train_test_split(self.data_gdf, test_size=(valid_prop+test_prop), train_size=train_prop, random_state=42, stratify=self.data_gdf['taxonID'])
        test, valid = ms.train_test_split(t_v, test_size=valid_prop/(valid_prop+test_prop), train_size=test_prop/(valid_prop+test_prop), random_state=42, stratify=t_v['taxonID'])
        return train, valid, test


    def save_valid_df(self, save_dir):
        class_columns = list(range(self.num_classes))
        base = ['species', 'expected']
        columns = base + class_columns
        df = pd.DataFrame(columns=columns)

        for key, value in self.valid_dict.items():
            species_dict = {}
            species_dict['species'] = key
            species_dict['expected'] = value['expected']
            for j, l in value['found'].items():
                if j < self.num_classes:
                    species_dict[j] = l
            df = pd.concat((df, pd.Series(species_dict).to_frame().T))

        df.to_csv(os.path.join(save_dir, 'valid_stats.csv'))
        return df

    @staticmethod
    def _extract_plot(df, orig_dict, save_dir):
        first_row = df.iloc[0]
        coords = first_row['file_coords']
        plot = first_row['plotID']
        rgb = hp.pre_processing(orig_dict[coords], wavelength_ranges=utils.get_viz_bands())
        rgb = hp.make_rgb(rgb["bands"])
        rgb = exposure.adjust_gamma(rgb, 0.5)
        rgb = rgb[first_row['y_min']:first_row['y_max'], first_row['x_min']:first_row['x_max'], :]
        plt.imsave(os.path.join(save_dir, f'{plot}.png'), rgb)

    def extract_plots(self, save_dir):
        grouped_files = self.valid_data.groupby(['plotID'])
        grouped_files.apply(self._extract_plot, self.orig_dict, save_dir)
        

    
    def extract_pca_plots(self, save_dir):
        grouped_files = self.valid_data.groupby(['plotID'])
        grouped_files.apply(self._extract_pca_plot, self.pca_dict, save_dir)

    @staticmethod
    def _extract_pca_plot(df, pca_dict, save_dir):
        first_row = df.iloc[0]
        coords = first_row['file_coords']
        plot = first_row['plotID']
        to_open = pca_dict[coords]
        pca_file = np.load(to_open)
        extract = pca_file[first_row['y_min']:first_row['y_max'], first_row['x_min']:first_row['x_max'], 0:3]
        mask = extract != extract
        extract[mask] = 0
        extract = (extract - np.min(extract))/np.ptp(extract)
        extract[mask] = 0


        plt.imsave(os.path.join(save_dir, f'{plot}.png'), extract)
        return df


    #TODO: Make generalizable, can use orig_dict
    @staticmethod
    def _map_plot(df, pca_dir, save_dir, site_name, prefix):
        fig, ax = plt.subplots(figsize=(10, 10))
        row = df.iloc[0]
        coords = row['file_coords']
        open_file = os.path.join(pca_dir, f'NEON_{prefix}_{site_name}_DP3_{coords}_reflectance.h5')
        rgb = hp.pre_processing(open_file, wavelength_ranges=utils.get_viz_bands())
        rgb = hp.make_rgb(rgb["bands"])
        rgb = exposure.adjust_gamma(rgb, 0.5)
        ax.imshow(rgb)
        x = [row['x_min'], row['x_max'], row['x_max'], row['x_min'], row['x_min']]
        y = [row['y_max'], row['y_max'], row['y_min'], row['y_min'], row['y_max']]
        ax.plot(x, y)
        ax.set_title(f'Original File: NEON_{prefix}_{site_name}_DP3_{coords}_reflectance.h5 \n PlotID: {row["plotID"]}')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'NEON_{prefix}_{site_name}_DP3_{coords}_Plot_{row["plotID"]}.png'))
        plt.close()

    
    @staticmethod
    def _select_pixels(df, vd):
        # fig, ax = plt.subplots(figsize=(10, 10))
        top_row = df.iloc[0]
        coords = top_row['file_coords']
        open_sp = vd.sp_dict[coords]
        open_chm = vd.chm_dict[coords]
        chm_open = rs.open(open_chm)
        chm = chm_open.read().astype(np.float32)
        chm[chm==-9999] = np.nan
        chm = chm.squeeze(axis=0)

        sp = np.load(open_sp)
        c_max = chm.max()
        chm_scale = chm/c_max


        trees_df = df.loc[(df['tree_x'] == df['tree_x'])]
        trees_df['sp'], trees_df['sp_height_dif'], trees_df['sp_east'], trees_df['sp_north'] = trees_df.apply(lambda row: vd._select_superpixel(row, sp, chm), axis=1).str
        trees_df = trees_df.loc[trees_df['sp'] != -1]

        if trees_df['sp'].duplicated().sum() != 0:
            for pix_num in trees_df['sp'].unique():
                to_dedupe = []
                testing = trees_df.loc[trees_df['sp'] == pix_num]
                if len(testing) > 1:
                    min_height = testing['sp_height_dif'].min()
                    #See if there are any with the same height difference
                    h_testing = testing.loc[testing['sp_height_dif'] == min_height]
                    #Remove any that aren't the minimum height difference
                    to_remove = testing.loc[testing['sp_height_dif'] != min_height]
                    to_dedupe= to_dedupe + list(to_remove.index)
                    #If there are any with the same height difference, we need to check what species they are
                    if len(h_testing) > 1:
                        unique_species = h_testing['taxonID'].unique()
                        if len(unique_species) == 1:
                            #If all the species are the same just pick the first one
                            to_dedupe = to_dedupe + list(h_testing.index[:-1])
                        else:
                            #If there are two species with the same height difference we don't have enough info to choose
                            to_dedupe= to_dedupe + list(h_testing.index)
                        

                    trees_df = trees_df.drop(to_dedupe)


        #Check for intersections
        check_inters =  gpd.GeoDataFrame(trees_df, geometry=gpd.points_from_xy(trees_df.sp_east, trees_df.sp_north), crs='EPSG:32618')
        crowns = check_inters.buffer(2)

        to_remove = set()
        for ix, row in crowns.iteritems():
            working_copy = crowns.loc[(crowns.index != ix)]
            intersect = working_copy.intersects(row)
            if intersect.sum() > 0:
                to_remove.add(ix)
        if len(to_remove)>0:
            trees_df =  trees_df.drop(list(to_remove))


        return trees_df

    @staticmethod
    def _select_superpixel(row, sp, chm, select_buffer=5, upper_height_bound=5):
        height = row['height']
        x = round(row['tree_x'])
        y = round(row['tree_y'])

        sp_crop = sp[y-select_buffer:y+select_buffer, x-select_buffer:x+select_buffer]


        unique_sp= np.unique(sp_crop)

        if unique_sp.sum() == 0:
            return -1, -1, -1, -1

        candidates = {}
        for pix_num in unique_sp:
            if pix_num == 0:
                continue
            chm_sp = chm[sp==pix_num]
            sp_height = chm_sp.max()
            height_dif = abs(sp_height - height)
            if height_dif <= upper_height_bound:
                candidates[pix_num] = height_dif
        
        if len(candidates.keys()) == 0:
            return -1, -1

        min_dif = 1000
        top_candidate = None
        for k, v in candidates.items():
            if v < min_dif:
                min_dif = v
                top_candidate = k

        if top_candidate is not None:
            #get coordinates of top candidate
            sp_mask = sp == top_candidate
            masked_chm = chm * sp_mask
            max_height = masked_chm.max()
            height_pix = masked_chm == max_height
            h_col = np.all(~height_pix, axis=0).argmin()
            h_row = np.all(~height_pix, axis=1).argmin()

            h_col = int(row['file_west_bound']) + h_col
            h_row = (int(row['file_south_bound']) + 1000) - h_row


            return top_candidate, min_dif, h_col, h_row
        else:
            return -1, -1, -1, -1


    def pick_superpixels(self):
        grouped_files = self.data_gdf.groupby(['file_coords', 'plotID'])
        selected = grouped_files.apply(self._select_pixels, self)
        return selected

    def do_plot_inference(self, save_dir, model):
        grouped_files = self.valid_data.groupby(['file_coords', 'plotID'])
        grouped_files.apply(self._plot_inference, self, save_dir, model)
        self.save_valid_df(save_dir)




    
    def map_plots(self, save_dir):
        grouped_files = self.valid_data.groupby(['file_coords', 'plotID'])
        grouped_files.apply(self._map_plot, self.orig_dir, save_dir, self.site_name, self.site_prefix)

    def make_taxa_plots(self, save_dir):
         grouped_files = self.valid_data.groupby(['file_coords', 'plotID'])
         grouped_files.apply(self._make_taxa_plot, save_dir)



    @staticmethod
    def _make_taxa_plot(df, save_dir):
        ax = df.plot.bar(x='taxonID', y='taxonCount')
        ax.set_title(df.iloc[0]['plotID'])
        plt.savefig(os.path.join(save_dir, f'{df.iloc[0]["plotID"]}_taxon_count.png'))

        return None    

    def make_taxa_area_hists(self, save_dir):
        grouped_files = self.valid_data.groupby(['file_coords', 'plotID'])
        grouped_files.apply(self._make_area_hist, save_dir)

    @staticmethod
    def _make_area_hist(df, save_dir):
        df = df.loc[df['approx_sq_m'] == df['approx_sq_m']]
        
        taxon_sums = df.groupby('taxonID').sum()
        taxon_sums = taxon_sums.reset_index()
        ax = taxon_sums.plot.bar(x='taxonID', y='ninetyCrownDiameter')
        ax.set_title(df.iloc[0]['plotID'])
        ax.set_ylabel('Pixel Count')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'{df.iloc[0]["plotID"]}_taxon_area.png'))

        sun_sums = df.groupby('sun').sum()
        sun_sums = sun_sums.reset_index()
        sun_sums['sun'].loc[sun_sums['sun'] == True] = 'Sun'
        sun_sums['sun'].loc[sun_sums['sun'] == False] = 'Shade'
        ax = sun_sums.plot.bar(x='sun', y='ninetyCrownDiameter')
        ax.set_title(df.iloc[0]['plotID'])
        ax.set_ylabel('Pixel Count')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'{df.iloc[0]["plotID"]}_sun_area.png'))




        
        print('here')

    def plot_tree(self, coord, file, **kwargs):
     
        fig, ax = plt.subplots()
        rgb = hp.pre_processing(file, wavelength_ranges=utils.get_viz_bands())
        rgb = hp.make_rgb(rgb["bands"])
        rgb = exposure.adjust_gamma(rgb, 0.6)
        rgb = exposure.rescale_intensity(rgb)
        
        ax.imshow(rgb)
        if 'predictions' in kwargs:
            loc = os.path.join(kwargs['predictions'], coord + '.npy')
            if os.path.isfile(loc):
                y = np.load(loc)
                #y[y!=52] = 0
                im = ax.imshow(y, alpha=.2)
                slider = self._make_slider(fig, im)
        data = self.valid_data.loc[self.valid_data['file_coords'] == coord]

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
            valstep=0.05,
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
            plots = valid["plotID"].unique()
            for plot in plots:
                valid_plot = valid.loc[valid['plotID'] == plot]
                row = valid_plot.iloc[0]
                select = clustered[row['y_min']:row['y_max'], row['x_min']:row['x_max']]
                groups, counts = np.unique(select, return_counts=True)
                self.cluster_groups = self.cluster_groups.union(set(groups.astype(int)))
                for j, group in np.ndenumerate(groups):
                    self.cluster_dict[plot]['found'][int(group)] = counts[j]
        return self.cluster_dict

    def make_empty_dict(self):
        #taxa = self.valid_data["taxonID"].unique()
        plots = self.valid_data['plotID'].unique()

        template = {plot:{'expected': {}, 'found': {}} for plot in plots}
        for plot in plots:
            valid = self.valid_data.loc[self.valid_data['plotID'] == plot]
            taxa = valid["taxonID"].unique()
            for taxon in taxa:

                template[plot]['expected'][taxon] = int(valid.loc[valid['taxonID'] == taxon]['approx_sq_m'])
        return template
    

    @property
    def confusion_matrix(self):
        reformed = {(key, i, k):l for key, value in self.cluster_dict.items() for i, j in value.items() for k, l in j.items()}
        mi = pd.MultiIndex.from_tuples(reformed.keys())
        mat = pd.DataFrame(list(reformed.values()), index=mi)
        

        return mat


    def kappa(self):
        return None
    



def side_by_side_bar(df, plot_name):
    fig, ax = plt.subplots(1, 2)
    cats = ('expected', 'found')
    for i, y in enumerate(ax):
        try:
            df.loc[plot_name, cats[i]].plot.bar(ax=y)
        except KeyError:
            continue
    plt.show()
    print('here')

def plot_species(validator: Validator, species):
    df = validator.valid_data
    groups = list(validator.cluster_groups)
    plots = df['plotID'].unique()
    
    combos = [(species, group) for group in groups]
    points = {combo:{'x':[], 'y':[]} for combo in combos}

    conf = validator.confusion_matrix

    for plot in plots:
        pdf = conf.loc[plot]
        for combo in combos:
            try: 
                expect = pdf.loc['expected']
                found = pdf.loc['found']
            except KeyError:
                continue
            if combo[0] in expect.index and combo[1] in found.index:
                points[combo]['x'].append(int(expect.loc[combo[0]]))
                points[combo]['y'].append(int(found.loc[combo[1]]))
    
    fig, ax = plt.subplots(6, 5, figsize=(15,15))
    ax = ax.flatten()
    for i, (combo, value) in enumerate(points.items()):
        if len(value['x'])>2:
            slope, intercept, r , p, se = linregress(value['x'], value['y'])
            ax[i].scatter(value['x'], value['y'])
            ax[i].set_title(f'{combo} r2: {r**2:.2f}')
    plt.tight_layout()
    plt.show()

def show_file(f):
    f = np.load(f)
    plt.imshow(f)
    plt.show()

def viz_and_save_plot(plot_dict, save_dir):
    selected = hp.get_selection(plot_dict['bands'], plot_dict['meta']['spectral_bands'], utils.get_viz_bands())
    rgb = hp.make_rgb(selected)
    rgb = exposure.adjust_gamma(rgb, 0.5)
    outname= os.path.join(save_dir, f'{plot_dict["meta"]["plotID"]}_{plot_dict["meta"]["original_file"]}.png')
    plt.imsave(outname, rgb)

def inc_pca_plots(plot_dir, save_dir):
    transformer = IncrementalPCA(n_components=10)
    for f in os.listdir(plot_dir):
        if ".pk" in f:
            with open(os.path.join(plot_dir, f), 'rb') as img:
                plot = pickle.load(img)
                all = plot['bands']
                all = rearrange(all, 'h w c -> (h w) c')
                transformer.fit(all)
    for f in os.listdir(plot_dir):
        if ".pk" in f:
            with open(os.path.join(plot_dir, f), 'rb') as img:
                plot = pickle.load(img)
                all = plot['bands']
                all = rearrange(all, 'h w c -> (h w) c')
                proc = transformer.transform(all)
                proc = rearrange(proc, '(h w) c -> h w c', h=40, w=40)
                np.save(os.path.join(save_dir, f), proc)
                to_img = (proc - np.min(proc))/np.ptp(proc)
                plt.imsave(os.path.join(save_dir,'first_three_viz', f"{f.split('.')[0]}.png"),to_img[...,0:3])

def ward_cluster_plots(plot_dir, save_dir):
    for f in os.listdir(plot_dir):
        if ".npy" in f:
            plot = np.load(os.path.join(plot_dir,f))
            plot = rearrange(plot, 'h w c -> (h w) c')
            clustered = utils.ward_cluster(plot, n_clusters=6)
            clustered = rearrange(clustered, '(h w) -> h w', h=40, w=40)
            np.save(os.path.join(save_dir, f'cluster_{f}'), clustered)
            plt.imsave(os.path.join(save_dir, 'viz',  f"{f.split('.')[0]}.png"),clustered)

def pca_norm_cluster_plots(plot_dir, save_dir):
    mean = np.load(os.path.join(plot_dir, 'stats/mean.npy')).astype(np.float32)
    std = np.load(os.path.join(plot_dir, 'stats/std.npy')).astype(np.float32)

    norm = tt.Normalize(mean, std)
    for f in os.listdir(plot_dir):
        if ".npy" in f:
            plot = np.load(os.path.join(plot_dir,f))
            plot = rearrange(plot, 'h w c -> c h w')
            img = torch.from_numpy(plot).float()
            #img = rearrange(img, 'h w c -> c h w')
            img = norm(img)
            mp = torch.nn.MaxPool2d(2)
            img = mp(img)
            up = torch.nn.UpsamplingBilinear2d(scale_factor=2)
            img = up(img.unsqueeze(0))
            img = torch.argmax(img.squeeze(0), dim=0)
            img = img.numpy()
            np.save(os.path.join(save_dir, f), img)
            plt.imsave(os.path.join(save_dir, 'viz',  f"{f.split('.')[0]}.png"),img)

def get_shadow_masks(plot_dict, save_dir):
    selected = hp.get_selection(plot_dict['bands'], plot_dict['meta']['spectral_bands'], utils.get_shadow_bands())
    rgb = hp.get_selection(plot_dict['bands'], plot_dict['meta']['spectral_bands'], utils.get_viz_bands())
    rgb = utils.make_rgb(rgb)
    mask = utils.han_2018(selected)
    rgb =exposure.adjust_gamma(rgb, 0.5)

    
    plot_dict['shadow_mask'] = mask

    return None

def get_spectra_plots(plot_dict, save_dir):

    wavelengths = plot_dict['meta']['spectral_bands']
    data = plot_dict['bands']

    data = rearrange(data, 'h w c -> (h w) c')

    mean = data.mean(axis=0)
    plt.plot(wavelengths, mean)
    plt.xlabel('Wavelength')
    plt.ylabel('Mean Value')
    plt.title(plot_dict['meta']['plotID'])
    plt.ylim(-0.01, 0.6)

    plt.savefig(os.path.join(save_dir, f'{plot_dict["meta"]["plotID"]}.png'))
    plt.close()

    return None

def cluster_histograms(plot_dir, save_dir):
    for f in os.listdir(plot_dir):
        if ".npy" in f:
            plot = np.load(os.path.join(plot_dir,f))
            plt.bar(*np.unique(plot, return_counts=True))
            plot_id = f.split("_")[2]+ " "+ f.split("_")[3]
            plt.title(plot_id)
            plt.xlabel('Classification')
            plt.ylabel('Pixel Count')
            plt.xlim(0, 10)

            plt.savefig(os.path.join(save_dir, plot_id + ".png"))
            plt.close()

                



def handle_each_plot(plot_dir, fn, save_dir):
    for f in os.listdir(plot_dir):
        if ".pk" in f:
            with open(os.path.join(plot_dir, f), 'rb') as img:
                plot = pickle.load(img)
                fn(plot, save_dir)

def make_valid_df(valid_dict, num_classes):
    class_columns = list(range(num_classes))
    base = ['species', 'expected']
    columns = base + class_columns
    df = pd.DataFrame(columns=columns)

    for key, value in valid_dict.items():
        species_dict = {}
        species_dict['species'] = key
        species_dict['expected'] = value['expected']
        for j, l in value['found'].items():
            if j < num_classes:
                species_dict[j] = l
        df = pd.concat((df, pd.Series(species_dict).to_frame().T))

    df.to_csv('test_kappa.csv')
    return df

# #TODO: add stat saving
# #Make models agnostic
# def validate_config(validator, config_list):
#     for config in config_list:
#         model = models.SWaVModelStruct(**config).load_from_checkpoint(config['ckpt'],**config)
#         try:
#             validator.last_cluster = {}
#             validator.do_plot_inference(config['save_dir'], model)
#         except KeyError as e:
#             print(f'Error: {e}')
#             continue



if __name__ == "__main__":
    NUM_CLASSES = 12
    NUM_CHANNELS = 10
    PCA_DIR= 'C:/Users/tonyt/Documents/Research/datasets/pca/niwo_16_unmasked'
   
    VALID_FILE = "W:/Classes/Research/neon-allsites-appidv-latest.csv"
    CURATED_FILE = "W:/Classes/Research/neon_niwo_mapped_struct_de_dupe.csv"
    PLOT_FILE = 'W:/Classes/Research/All_NEON_TOS_Plots_V8/All_NEON_TOS_Plots_V8/All_NEON_TOS_Plot_Centroids_V8.csv'
    CHM_DIR = 'C:/Users/tonyt/Documents/Research/datasets/chm/niwo'
    AZM_DIR = 'C:/Users/tonyt/Documents/Research/datasets/solar_azimuth/niwo/'
    ORIG_DIR = 'W:/Classes/Research/datasets/hs/original/NEON.D13.NIWO.DP3.30006.001.2020-08.basic.20220516T164957Z.RELEASE-2022'
    ICA_DIR = 'C:/Users/tonyt/Documents/Research/datasets/ica/niwo_10_channels'
    RAW_DIR = 'C:/Users/tonyt/Documents/Research/datasets/selected_bands/niwo/all'
    SHADOW_DIR = 'C:/Users/tonyt/Documents/Research/datasets/mpsi/niwo'
    SP_DIR = 'C:/Users/tonyt/Documents/Research/datasets/superpixels/niwo_chm'
    INDEX_DIR = 'C:/Users/tonyt/Documents/Research/datasets/indexes/niwo/'

    valid = Validator(file=VALID_FILE, 
                    pca_dir=PCA_DIR, 
                    ica_dir=ICA_DIR,
                    raw_bands=RAW_DIR,
                    shadow=SHADOW_DIR,
                    site_name='NIWO', 
                    num_classes=NUM_CLASSES, 
                    plot_file=PLOT_FILE, 
                    struct=True, 
                    azm=AZM_DIR, 
                    chm=CHM_DIR, 
                    curated=CURATED_FILE, 
                    rescale=False, 
                    orig=ORIG_DIR, 
                    superpixel=SP_DIR,
                    indexes=INDEX_DIR,
                    prefix='D13',
                    chm_mean = 4.015508459469479,
                    chm_std = 4.809300736115787,
                    use_sp=False,
                    scholl_filter=True,
                    scholl_output=True,
                    filter_species = 'SALIX')


    valid.render_valid_patch('C:/Users/tonyt/Documents/Research/datasets/tensors/niwo_2020_pca_blocks/scholl_train_mc', 'train', out_size=20, multi_crop=10)
    # valid.render_valid_patch('C:/Users/tonyt/Documents/Research/datasets/tensors/niwo_2020_nmf_blocks/scholl_valid', 'valid', out_size=20)
    # valid.render_valid_patch('C:/Users/tonyt/Documents/Research/datasets/tensors/niwo_2020_nmf_blocks/scholl_test', 'test', out_size=20)
    print(valid.taxa)
    print(valid.class_weights)


