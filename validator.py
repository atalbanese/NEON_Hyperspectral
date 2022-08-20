from select import select
from typing_extensions import Self
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
    def __init__(self, train_split=0.6, valid_split=0.2, test_split=0.2, use_tt=True, scholl_filter=False, scholl_output=False, filter_species=None, object_split=False, tree_tops_dir="", data_gdf=None, **kwargs):
        self.file = kwargs["file"]
        self.pca_dir = kwargs["pca_dir"]
        self.tree_tops_dir = tree_tops_dir
        self.curated = kwargs['curated']
        self.site_prefix = kwargs['prefix']

        
        self.num_classes = kwargs["num_classes"]
        self.site_name = kwargs["site_name"]
        self.plot_file = kwargs['plot_file']
        self.orig_dir = kwargs['orig']
        self.use_tt = use_tt
        self.scholl_filter = scholl_filter
        self.scholl_output = scholl_output
        self.filter_species = filter_species

        self.data_gdf = self.get_plot_data()

        self.valid_files = self.get_valid_files()

        self.valid_dict = self.make_valid_dict()
        
        

        self.orig_files = [os.path.join(kwargs['orig'], file) for file in os.listdir(kwargs['orig']) if ".h5" in file]
        self.pca_files = [os.path.join(kwargs['pca_dir'], file) for file in os.listdir(kwargs['pca_dir']) if ".npy" in file]
        self.tree_tops_files = [os.path.join(tree_tops_dir, file) for file in os.listdir(tree_tops_dir) if ".geojson" in file]
        
        def make_dict(file_list, param_1, param_2):
            return {f"{file.split('_')[param_1]}_{file.split('_')[param_2]}": file for file in file_list}

        self.orig_dict = make_dict(self.orig_files, -3, -2)
        self.pca_dict = make_dict(self.pca_files, -4, -3)
        self.tree_tops_dict = make_dict(self.tree_tops_files, -3, -2)

        
        self.rng = np.random.default_rng(42)
        self.taxa = {key: ix for ix, key in enumerate(self.data_gdf['taxonID'].unique())}
        
        if use_tt:
            if data_gdf is None:
                self.data_gdf = self.pick_ttops()
                with open('test_gdf.pkl', 'wb') as f:
                    pickle.dump(self.data_gdf, f)
            else:
                with open(data_gdf, 'rb') as f:
                    self.data_gdf = pickle.load(f)
        

        # self.data_gdf = gpd.GeoDataFrame(self.data_gdf, geometry=gpd.points_from_xy(self.data_gdf.easting_tree, self.data_gdf.northing_tree), crs='EPSG:32613')
        # self.data_gdf.geometry = self.data_gdf.buffer(self.data_gdf['maxCrownDiameter']*0.9/2)

        self.valid_plots, self.train_plots, self.test_plots = self.split_plots().values()

        if object_split:
            self.train, self.valid, self.test = self.get_splits(train_split, valid_split, test_split)
        else:
            self.train = self.data_gdf.loc[self.data_gdf.plotID.isin(list(self.train_plots.index))]
            self.valid = self.data_gdf.loc[self.data_gdf.plotID.isin(list(self.valid_plots.index))]
            self.test = self.data_gdf.loc[self.data_gdf.plotID.isin(list(self.test_plots.index))]
        self.class_weights = cw.compute_class_weight(class_weight='balanced', classes=self.data_gdf['taxonID'].unique(), y=self.train['taxonID'])

    def save_splits(self, save_dir):
        to_save = {'train': self.train,
                    'test': self.test,
                    'valid': self.valid}
        for k, v in to_save.items():
            save_loc = os.path.join(save_dir, f'{k}.shp')
            v = v[['taxonID', 'geometry']]
            v['taxonID'] = v['taxonID'].astype(str)
            v.to_file(save_loc)

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



    def get_splits(self, train_prop, valid_prop, test_prop):
        train, t_v = ms.train_test_split(self.data_gdf, test_size=(valid_prop+test_prop), train_size=train_prop, random_state=42, stratify=self.data_gdf['taxonID'])
        test, valid = ms.train_test_split(t_v, test_size=valid_prop/(valid_prop+test_prop), train_size=test_prop/(valid_prop+test_prop), random_state=42, stratify=t_v['taxonID'])
        return train, valid, test

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


            if not self.use_tt:
                t_col = int(row['tree_x'])
                t_row = int(row['tree_y'])
            else:
                t_col = round(row.geometry.centroid.x) - int(row.file_west_bound)
                t_row = 1000 - (round(row.geometry.centroid.y) - int(row.file_south_bound))



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
            
        data_gdf = gpd.GeoDataFrame(data_with_coords, geometry=gpd.points_from_xy(data_with_coords.easting, data_with_coords.northing, data_with_coords.height), crs='EPSG:32613')
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
        #data_gdf = pd.DataFrame(data_gdf)

        return data_gdf



    def get_valid_files(self):
        if self.pca_dir is not None:
            coords = list(self.data_gdf['file_coords'].unique())

            all_files = os.listdir(self.pca_dir)
            valid_files = {coord:os.path.join(self.pca_dir,file) for file in all_files for coord in coords if coord in file}
            return valid_files
        else:
            return None


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

    
    def split_plots(self, splits={'valid':0.2,  'train': 0.6, 'test':0.2}):
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

        if len(counts) != 0:
            solved['test'] = pd.concat((solved['test'], counts), axis=0)

        return solved

    
    @staticmethod
    def _split_plot(df: pd.DataFrame, pc: dict):
        if len(df) > 0:
            #counts = df.groupby(['taxonID']).size()
            pc[df.iloc[0]['plotID']] = {tx:0 for tx in df.taxonID.unique()}

            for tx in df.taxonID.unique():
                cur = df.loc[df.taxonID == tx]
                area= cur.area.sum()
                pc[df.iloc[0]['plotID']][tx] = area




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
    def _select_ttops(df, vd):
        top_row = df.iloc[0]
        coords = top_row['file_coords']
        open_ttops = gpd.read_file(vd.tree_tops_dict[coords])

        df['cur_buffer'] = df.buffer(5)

        clip = open_ttops.clip(df.cur_buffer)

        #TODO: Switch to spatial index

        def find_match(row, clip):
            distances = clip.distance(row.geometry)
            distances = distances.loc[distances<5]
            if len(distances) > 0:
                distances = distances.sort_values()
                return distances.index[0], distances.iloc[0]
            else:
                return -1, -1

        df['best_match'], df['match_distance'] = df.apply(lambda row: find_match(row, clip), axis=1).str

        

        df = df.loc[df['best_match'] != -1]
        if df['best_match'].duplicated().sum() != 0:
            to_dedupe = []
            for pix_num in df['best_match'].unique():
                
                testing = df.loc[df['best_match'] == pix_num]
                if len(testing) > 1:
                    min_dist = testing['match_distance'].min()
                    to_remove = testing.loc[testing['match_distance'] > min_dist]
                    to_dedupe = to_dedupe + list(to_remove.index)

                    dist_match = testing.loc[testing['match_distance'] == min_dist]
                    if len(dist_match) > 1:
                        unique_species = dist_match['taxonID'].unique()
                        if len(unique_species) == 1:
                            to_dedupe = to_dedupe + list(dist_match.index[:-1])
                        else:
                            to_dedupe = to_dedupe + list(dist_match.index)
            df = df.drop(to_dedupe)

        #Work around for just resetting index since that drops geometry
        select = clip.loc[df.best_match]
        select.index = df.index
        df.geometry = select.geometry

        df.geometry = df.buffer(df['maxCrownDiameter']*0.9/2)
        if len(df) >0:
            df['taxa_key'] = df.apply(lambda x: vd.taxa[x.taxonID], axis=1)
            # fig, ax = plt.subplots(1,1)
            # df.plot(column='taxa_key', ax=ax, legend=True)
            # plt.show()

            to_de_intersect = []
            #df.crown_buffer = df.buffer(df['height'])
            df = df.sort_values('height', ascending=False)
            for ix, row in df.iterrows():
                working = df.loc[df.index != ix]
                clipper = df.loc[df.index == ix]
                if len(working) > 1:
                    inter = working.clip(clipper)
                    if len(inter)> 0:
                        all = pd.concat((inter, clipper))
                        #Only remove intersections when species doesn't match
                        if len(all.taxonID.unique())> 1:
                            all = all.sort_values('height', ascending=False)
                            top_taxon = all.iloc[0].taxonID
                            to_remove = all.loc[all.taxonID != top_taxon]
                            to_de_intersect = to_de_intersect + list(to_remove.index)

            df = df.drop(to_de_intersect)


        return df
   

    def pick_ttops(self):
        grouped_files = self.data_gdf.groupby('plotID')
        selected = grouped_files.apply(self._select_ttops, self)
        selected = selected.drop('plotID', axis=1).reset_index()
        return selected




    
    def map_plots(self, save_dir):
        grouped_files = self.data_gdf.groupby(['file_coords', 'plotID'])
        grouped_files.apply(self._map_plot, self.orig_dir, save_dir, self.site_name, self.site_prefix)




        

    





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
    TREE_TOPS_DIR = 'C:/Users/tonyt/Documents/Research/datasets/lidar/niwo_point_cloud/valid_sites_ttops'

    valid = Validator(file=VALID_FILE, 
                    pca_dir=PCA_DIR, 
                    site_name='NIWO', 
                    num_classes=NUM_CLASSES, 
                    plot_file=PLOT_FILE, 
                    tree_tops_dir=TREE_TOPS_DIR,
                    curated=CURATED_FILE, 
                    orig=ORIG_DIR, 
                    prefix='D13',
                    use_tt=True,
                    scholl_filter=False,
                    scholl_output=False,
                    filter_species = 'SALIX',
                    object_split=False,
                    data_gdf='test_gdf.pkl')
    valid.save_splits('C:/Users/tonyt/Documents/Research/split_jsons/ttops_plot_split')
    #print(valid.valid_files.keys())


   
    # valid.render_plots('C:/Users/tonyt/Documents/Research/datasets/tensors/niwo_2020_pca_blocks/full_plot_train', 'train', filetype='pca')
    # valid.render_plots('C:/Users/tonyt/Documents/Research/datasets/tensors/niwo_2020_pca_blocks/full_plot_test', 'test', filetype='pca')
    # valid.render_plots('C:/Users/tonyt/Documents/Research/datasets/tensors/niwo_2020_pca_blocks/full_plot_valid', 'valid', filetype='pca')
    valid.render_valid_patch('C:/Users/tonyt/Documents/Research/datasets/tensors/niwo_2020_pca_blocks/tt_train_hs', 'train', out_size=20, key_label='hs', num_channels=416, filters=['ndvi', 'shadow'])
    valid.render_valid_patch('C:/Users/tonyt/Documents/Research/datasets/tensors/niwo_2020_pca_blocks/tt_valid_hs', 'valid', out_size=20, key_label='hs', num_channels=416, filters=['ndvi', 'shadow'])
    valid.render_valid_patch('C:/Users/tonyt/Documents/Research/datasets/tensors/niwo_2020_pca_blocks/tt_test_hs', 'test', out_size=20, key_label='hs', num_channels=416, filters=['ndvi', 'shadow'])
    valid.render_valid_patch('C:/Users/tonyt/Documents/Research/datasets/tensors/niwo_2020_pca_blocks/tt_train', 'train', out_size=20, key_label='pca', num_channels=16, filters=['ndvi', 'shadow'])
    valid.render_valid_patch('C:/Users/tonyt/Documents/Research/datasets/tensors/niwo_2020_pca_blocks/tt_valid', 'valid', out_size=20, key_label='pca', num_channels=16, filters=['ndvi', 'shadow'])
    valid.render_valid_patch('C:/Users/tonyt/Documents/Research/datasets/tensors/niwo_2020_pca_blocks/tt_test', 'test', out_size=20, key_label='pca', num_channels=16, filters=['ndvi', 'shadow'])

    # valid_2 = Validator(file=VALID_FILE, 
    #             pca_dir=PCA_DIR, 
    #             ica_dir=ICA_DIR,
    #             raw_bands=RAW_DIR,
    #             shadow=SHADOW_DIR,
    #             site_name='NIWO', 
    #             num_classes=NUM_CLASSES, 
    #             plot_file=PLOT_FILE, 
    #             struct=True, 
    #             azm=AZM_DIR, 
    #             chm=CHM_DIR, 
    #             curated=CURATED_FILE, 
    #             rescale=False, 
    #             orig=ORIG_DIR, 
    #             superpixel=SP_DIR,
    #             indexes=INDEX_DIR,
    #             prefix='D13',
    #             chm_mean = 4.015508459469479,
    #             chm_std = 4.809300736115787,
    #             use_sp=False,
    #             scholl_filter=True,
    #             scholl_output=False,
    #             filter_species = 'SALIX')

    # valid_2.render_valid_patch('C:/Users/tonyt/Documents/Research/datasets/tensors/niwo_2020_pca_blocks/scholl_train', 'train', out_size=20, key_label='pca', num_channels=16, filters=['ndvi', 'shadow'])
    # valid_2.render_valid_patch('C:/Users/tonyt/Documents/Research/datasets/tensors/niwo_2020_pca_blocks/scholl_valid', 'valid', out_size=20, key_label='pca', num_channels=16, filters=['ndvi', 'shadow'])
    # valid_2.render_valid_patch('C:/Users/tonyt/Documents/Research/datasets/tensors/niwo_2020_pca_blocks/scholl_test', 'test', out_size=20, key_label='pca', num_channels=16, filters=['ndvi', 'shadow'])


    print(valid.taxa)
    print(valid.class_weights)

    # print(valid_2.taxa)
    # print(valid_2.class_weights)


