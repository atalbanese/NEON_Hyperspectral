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

        self.data_gdf = self.get_plot_data()

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

        self.train_plots, self.valid_plots, self.test_plots = self.split_plots().values()

        #self.train, self.valid, self.test = self.get_splits(train_split, valid_split, test_split)
        self.train = self.data_gdf.loc[self.data_gdf.plotID.isin(list(self.train_plots.index))]
        self.valid = self.data_gdf.loc[self.data_gdf.plotID.isin(list(self.valid_plots.index))]
        self.test = self.data_gdf.loc[self.data_gdf.plotID.isin(list(self.test_plots.index))]
        self.class_weights = cw.compute_class_weight(class_weight='balanced', classes=self.data_gdf['taxonID'].unique(), y=self.train['taxonID'])

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


    def render_hs_data(self, save_dir, split, out_size=3, target_size=3):
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
               
                orig = hp.pre_processing(os.path.join(self.orig_dict[key]), get_all=True)["bands"][:,:,5:-5]
                bad_mask = np.zeros((1000,1000), dtype=bool)
                for i in range(0, orig.shape[-1]):
                    z = orig[:,:,i]
                    y = z>1
                    bad_mask += y
                orig[bad_mask] = np.nan
            
            

            taxa = row['taxonID']
            
            height = float(row['height'])


            rad = out_size//2 

            t_col = int(row['tree_x'])
            t_row = int(row['tree_y'])
            #LOOKUP HOW I USED TO DO THIS
            bounds = (t_row - rad, t_row+rad+1, t_col-rad, t_col+rad+1)
                
            orig_crop = orig[bounds[0]:bounds[1], bounds[2]:bounds[3],:]


            mask = orig_crop != orig_crop

            if orig_crop.shape == (out_size, out_size, orig.shape[-1]):
                orig_crop = torch.tensor(orig_crop)
                orig_crop = rearrange(orig_crop, 'h w c -> c h w')

                label = torch.zeros((len(self.taxa.keys()),target_size, target_size), dtype=torch.float32).clone()
                label[self.taxa[taxa]] = 1.0

                to_save = {
                    'orig': orig_crop,
                    'mask': mask,
                    'target': label,
                    'height': height,
                }

                f_name = f'{key}_{row["taxonID"]}_{row["individualID"]}.pt'
                with open(os.path.join(save_dir, f_name), 'wb') as f:
                    torch.save(to_save, f)
            
        return None

    def render_plots(self, save_dir, split, filetype='pca'):
        if split == 'train':
            data = self.train
        if split == 'valid':
            data = self.valid
        if split == 'test':
            data = self.test
        grouped_files = data.groupby(['plotID'])
        grouped_files.apply(self._render_plot, save_dir, self, filetype)
    
    @staticmethod
    def _render_plot(df: pd.DataFrame, save_dir: str, vd, filetype):
        first_row = df.iloc[0]
        f_key = first_row['file_coords']
        plot_id = first_row['plotID']
        if filetype == 'pca':
            pca = np.load(vd.pca_dict[f_key]).astype(np.float32)
        if filetype == 'hs':
            pca = hp.pre_processing(os.path.join(vd.orig_dict[f_key]), get_all=True)["bands"][:,:,5:-5]
        taxa = vd.taxa

        plot_bounds = (first_row['y_min'], first_row['y_max'], first_row['x_min'], first_row['x_max'])
        pca_plot = pca[plot_bounds[0]:plot_bounds[1], plot_bounds[2]:plot_bounds[3], ...]
        pca_plot = rearrange(pca_plot, 'h w c -> c h w')
        pca_plot = torch.from_numpy(pca_plot).to(torch.float32)

        key = torch.zeros((pca.shape[0], pca.shape[1]), dtype=torch.float32)

        for ix, row in df.iterrows():
            taxa_num = taxa[row['taxonID']] + 1
            tree_x = int(row['tree_x'])
            tree_y = int(row['tree_y'])
            radius = int(row['maxCrownDiameter']//2)

            key[tree_y-radius:tree_y+radius+1, tree_x-radius:tree_x+radius+1] = taxa_num

        key = key[plot_bounds[0]:plot_bounds[1], plot_bounds[2]:plot_bounds[3]]
        
        mask = key == 0
        key = key - 1
        #key[mask] = np.nan

        to_save = {'pca': pca_plot,
                    'targets': key,
                    'mask': mask}

        f_name = f'{f_key}_{plot_id}.pt'
        with open(os.path.join(save_dir, f_name), 'wb') as f:
            torch.save(to_save, f)

        return None




    def render_valid_patch(self, save_dir, split, out_size=20, multi_crop=1, num_channels=16, key_label='pca', filters=[]):
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
                if key_label == 'pca':
                    pca = np.load(self.pca_dict[key]).astype(np.float32)
                if key_label == 'hs':
                    pca = hp.pre_processing(os.path.join(self.orig_dict[key]), get_all=True)["bands"][:,:,5:-5]
                    bad_mask = np.zeros((1000,1000), dtype=bool)
                    for i in range(0, pca.shape[-1]):
                        z = pca[:,:,i]
                        y = z>1
                        bad_mask += y
                    pca[bad_mask] = np.nan
            
            
            rad = int(row['maxCrownDiameter']//2)
            target_size = rad*2 + 1

            taxa = row['taxonID']
            
            height = float(row['height'])


            if not self.use_sp:
                t_col = int(row['tree_x'])
                t_row = int(row['tree_y'])
            else:
                t_col = int(row['sp_east']) - int(row['file_west_bound'])
                t_row = 1000 - (int(row['sp_north'])- int(row['file_south_bound']))

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

                #crop_coords = (neg_y_pad, neg_x_pad, 3, 3)
                       
                if pca_crop.shape == (out_size, out_size, num_channels):
                    pca_crop = torch.tensor(pca_crop)
                    pca_crop = rearrange(pca_crop, 'h w c -> c h w')
                    masks = {}
                    if 'ndvi' in filters:
                        ndvi = hp.pre_processing(os.path.join(self.orig_dict[key]), utils.get_bareness_bands())["bands"]
                        
                        ndvi = utils.get_ndvi(ndvi)
                        ndvi = ndvi[y_min:y_max, x_min:x_max]
                        ndvi_mask = ndvi < 0.5
                        masks['ndvi'] = ndvi_mask
                    if 'shadow' in filters:
                        shadow = hp.pre_processing(os.path.join(self.orig_dict[key]), utils.get_shadow_bands())["bands"]
                        
                        shadow = utils.han_2018(shadow)
                        shadow = shadow[y_min:y_max, x_min:x_max]
                        shadow_mask = shadow < 0.03
                        masks['shadow'] = shadow_mask
                    #pca_crop[mask] = 0
                    

                    label = torch.zeros((out_size, out_size), dtype=torch.float32).clone()
                    #label[self.taxa[taxa]] = 1.0
                    label = label - 1
                    label[neg_y_pad:neg_y_pad+target_size+1, neg_x_pad:neg_x_pad+target_size+1] = self.taxa[taxa]

                    to_save = {
                        key_label: pca_crop,
                        'mask': masks,
                        'targets': label,
                        'height': height,
                        #'crop_coords': crop_coords
                    }

                    f_name = f'{key}_{row["taxonID"]}_{row["individualID"]}_multi_crop_{ix}.pt'
                    with open(os.path.join(save_dir, f_name), 'wb') as f:
                        torch.save(to_save, f)
            
        return None


    def make_valid_dict(self):
        species = self.data_gdf['taxonID'].unique()
        valid_dict = {}
        for specie in species:
            valid_dict[specie] = {'expected':0,
                                 'found': {i:0 for i in range(self.num_classes)}}
        return valid_dict



        

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

        data_gdf = data_gdf.merge(plots, how='left', on='plotID')

        data_gdf = data_gdf.rename(columns={
                                    'easting_x': 'easting_tree',
                                    'northing_x': 'northing_tree',
                                    'easting_y': 'easting_plot',
                                    'northing_y': 'northing_plot'
        })

        data_gdf["file_west_bound"] = data_gdf["easting_plot"] - data_gdf["easting_plot"] % 1000
        data_gdf["file_south_bound"] = data_gdf["northing_plot"] - data_gdf["northing_plot"] % 1000

        data_gdf = data_gdf.loc[data_gdf['file_west_bound'] == data_gdf['file_west_bound']]

        data_gdf = data_gdf.astype({"file_west_bound": int,
                            "file_south_bound": int})

        data_gdf['x_min'] = (data_gdf['easting_plot']//1 - data_gdf['file_west_bound']) - (data_gdf['plotSize']**(1/2)/2)
        data_gdf['x_max'] = data_gdf['x_min'] + data_gdf['plotSize']**(1/2)

        data_gdf['y_min'] = 1000- (data_gdf['northing_plot']//1 - data_gdf['file_south_bound']) - (data_gdf['plotSize']**(1/2)/2)
        data_gdf['y_max'] = data_gdf['y_min'] + data_gdf['plotSize']**(1/2)

        data_gdf['tree_x'] = data_gdf['easting_tree']//1 - data_gdf['file_west_bound']
        data_gdf['tree_y'] = 1000 - (data_gdf['northing_tree']//1 - data_gdf['file_south_bound'])

        data_gdf = data_gdf.astype({"file_west_bound": str,
                            "file_south_bound": str,
                            'x_min':int,
                            'y_min':int,
                            'x_max':int,
                            'y_max': int})
        
        index_names = data_gdf[(data_gdf['x_min'] <0) | (data_gdf['y_min']<0) | (data_gdf['x_max'] >999) | (data_gdf['y_max']>999)].index
        data_gdf = data_gdf.drop(index_names)

        data_gdf['file_coords'] = data_gdf['file_west_bound'] + '_' + data_gdf['file_south_bound']
        data_gdf = pd.DataFrame(data_gdf)

        return data_gdf



    def get_valid_files(self):
        if self.pca_dir is not None:
            coords = list(self.data_gdf['file_coords'].unique())

            all_files = os.listdir(self.pca_dir)
            valid_files = {coord:os.path.join(self.pca_dir,file) for file in all_files for coord in coords if coord in file}
            return valid_files
        else:
            return None




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

    @staticmethod
    def solve_split(df, prop, buffer):

        tracking = {k:False for k in df.columns}
        solved = False

        working_copy = df.iloc[:0].copy()

        while not solved:
            if len(df) < 1:
                break
            for k, v in tracking.items():
                if len(df) < 1:
                    break
                if v == False:
                    df = df.sort_values(k, ascending=False)
                    row = pd.DataFrame(df.iloc[0]).T
                    good_row = True
                    for j, i in tracking.items():
                        if row[j].item() + working_copy[j].sum() > buffer + prop:
                            good_row = False
                    df = df.drop(df.iloc[0].name)

                    if good_row:
                        working_copy = pd.concat((working_copy, row), axis=0)

                m_sum = 0
                for l, m in tracking.items():

                    if not m:
                        if working_copy[l].sum() > (prop - buffer):
                            tracking[l] = True
                            continue
                    else:
                        m_sum += m
                        if m_sum == len(tracking.values()):
                            solved = True
        distance = 0
        for k in tracking.keys():
            distance += abs(working_copy[k].sum() - prop)**2
        
        return {'solution':working_copy, 'dist':distance}

    
    def split_plots(self, splits={'train': 0.6, 'valid':0.2, 'test':0.2}):
        plot_counts={}
        grouped_files = self.data_gdf.groupby(['plotID'])
        grouped_files.apply(self._split_plot, plot_counts)
        counts = pd.DataFrame.from_dict(plot_counts).T
        for col in counts.columns:
            counts[col] = counts[col]/counts[col].sum()

        solved = {}
        for k, v in splits.items():
            cur_solutions = {}
            

            for buffer in range(1, 50):
                buffer /= 100
                cur_solutions[buffer] = self.solve_split(counts, v, buffer) 

            best_distance = 1000
            
            for buffer, solution in cur_solutions.items():
                if solution['dist'] < best_distance:
                    best_distance = solution['dist']
                    best_solution = solution['solution']

            solved[k] = best_solution
            counts = counts.drop(best_solution.index)


        return solved

    
    @staticmethod
    def _split_plot(df: pd.DataFrame, pc: dict):
        if len(df) > 0:
            counts = df.groupby(['taxonID']).size()
            pc[df.iloc[0]['plotID']] = counts



    def extract_plots(self, save_dir):
        grouped_files = self.data_gdf.groupby(['plotID'])
        grouped_files.apply(self._extract_plot, self.orig_dict, save_dir)
        

    
    def extract_pca_plots(self, save_dir):
        grouped_files = self.data_gdf.groupby(['plotID'])
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
        selected = selected.drop(['file_coords', 'plotID'], axis=1).reset_index()
        return selected




    
    def map_plots(self, save_dir):
        grouped_files = self.data_gdf.groupby(['file_coords', 'plotID'])
        grouped_files.apply(self._map_plot, self.orig_dir, save_dir, self.site_name, self.site_prefix)




        




        #for ix, row in self.data_gdf.iterrows():


    def make_empty_dict(self):
        #taxa = self.data_gdf["taxonID"].unique()
        plots = self.data_gdf['plotID'].unique()

        template = {plot:{'expected': {}, 'found': {}} for plot in plots}
        for plot in plots:
            valid = self.data_gdf.loc[self.data_gdf['plotID'] == plot]
            taxa = valid["taxonID"].unique()
            for taxon in taxa:

                template[plot]['expected'][taxon] = int(valid.loc[valid['taxonID'] == taxon]['approx_sq_m'])
        return template
    





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
                    use_sp=True,
                    scholl_filter=False,
                    scholl_output=False,
                    filter_species = 'SALIX')


   
    # valid.render_plots('C:/Users/tonyt/Documents/Research/datasets/tensors/niwo_2020_pca_blocks/full_plot_train', 'train', filetype='pca')
    # valid.render_plots('C:/Users/tonyt/Documents/Research/datasets/tensors/niwo_2020_pca_blocks/full_plot_test', 'test', filetype='pca')
    # valid.render_plots('C:/Users/tonyt/Documents/Research/datasets/tensors/niwo_2020_pca_blocks/full_plot_valid', 'valid', filetype='pca')
    valid.render_valid_patch('C:/Users/tonyt/Documents/Research/datasets/tensors/niwo_2020_pca_blocks/sp_train_hs', 'train', out_size=20, key_label='hs', num_channels=416, filters=['ndvi', 'shadow'])
    valid.render_valid_patch('C:/Users/tonyt/Documents/Research/datasets/tensors/niwo_2020_pca_blocks/sp_valid_hs', 'valid', out_size=20, key_label='hs', num_channels=416, filters=['ndvi', 'shadow'])
    valid.render_valid_patch('C:/Users/tonyt/Documents/Research/datasets/tensors/niwo_2020_pca_blocks/sp_test_hs', 'test', out_size=20, key_label='hs', num_channels=416, filters=['ndvi', 'shadow'])
    valid.render_valid_patch('C:/Users/tonyt/Documents/Research/datasets/tensors/niwo_2020_pca_blocks/sp_train', 'train', out_size=20, key_label='pca', num_channels=16, filters=['ndvi', 'shadow'])
    valid.render_valid_patch('C:/Users/tonyt/Documents/Research/datasets/tensors/niwo_2020_pca_blocks/sp_valid', 'valid', out_size=20, key_label='pca', num_channels=16, filters=['ndvi', 'shadow'])
    valid.render_valid_patch('C:/Users/tonyt/Documents/Research/datasets/tensors/niwo_2020_pca_blocks/sp_test', 'test', out_size=20, key_label='pca', num_channels=16, filters=['ndvi', 'shadow'])

    valid_2 = Validator(file=VALID_FILE, 
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
                scholl_output=False,
                filter_species = 'SALIX')

    valid_2.render_valid_patch('C:/Users/tonyt/Documents/Research/datasets/tensors/niwo_2020_pca_blocks/scholl_train', 'train', out_size=20, key_label='pca', num_channels=16, filters=['ndvi', 'shadow'])
    valid_2.render_valid_patch('C:/Users/tonyt/Documents/Research/datasets/tensors/niwo_2020_pca_blocks/scholl_valid', 'valid', out_size=20, key_label='pca', num_channels=16, filters=['ndvi', 'shadow'])
    valid_2.render_valid_patch('C:/Users/tonyt/Documents/Research/datasets/tensors/niwo_2020_pca_blocks/scholl_test', 'test', out_size=20, key_label='pca', num_channels=16, filters=['ndvi', 'shadow'])


    print(valid.taxa)
    print(valid.class_weights)

    print(valid_2.taxa)
    print(valid_2.class_weights)


