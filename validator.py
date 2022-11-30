from sklearn.preprocessing import StandardScaler
import skimage.segmentation as skseg 

import geopandas as gpd

import rasterio as rs
from rasterio.transform import from_origin
import rasterio.features as rf
import torch
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import h5_helper as hp
import os
import utils
from skimage import exposure

from einops import rearrange
import sklearn.model_selection as ms
import sklearn.utils.class_weight as cw
from shapely.geometry import Polygon
from rasterio.transform import from_origin



#TODO: Clean this mess up
#Switch plots to plotly? Good for overlays
class Validator():
    def __init__(self, 
                train_split=0.6, 
                valid_split=0.2, 
                test_split=0.2, 
                use_tt=True, 
                scholl_filter=False, 
                scholl_output=False, 
                filter_species=None, 
                object_split=False, 
                tree_tops_dir="", 
                rgb_dir = '',
                stats_loc = '',
                data_gdf=None, 
                ttops_search_buffer=5, 
                crown_diameter_mult = 0.9,
                constant_diameter = 2,
                ndvi_filter=None,
                **kwargs):
        self.file = kwargs["file"]
        self.pca_dir = kwargs["pca_dir"]
        self.tree_tops_dir = tree_tops_dir
        self.rgb_dir = rgb_dir
        self.stats_loc = stats_loc
        self.curated = kwargs['curated']
        self.site_prefix = kwargs['prefix']
        self.crown_diameter_mult = crown_diameter_mult
        self.constant_diameter = constant_diameter

        self.ndvi_filter=ndvi_filter
        self.num_classes = kwargs["num_classes"]
        self.site_name = kwargs["site_name"]
        self.plot_file = kwargs['plot_file']
        self.orig_dir = kwargs['orig']
        self.use_tt = use_tt
        self.scholl_filter = scholl_filter
        self.scholl_output = scholl_output
        self.filter_species = filter_species
        self.ttops_search_buffer = ttops_search_buffer

        self.data_gdf = self.get_plot_data()

        self.valid_files = self.get_valid_files()

        self.valid_dict = self.make_valid_dict()
        self.species_lookup = {'PICOL': 'Lodgepole Pine',
                                'PIFL2': 'Limber Pine',
                                'ABLAL': 'Subalpine Fir',
                                'PIEN': 'Engelmann Spruce'}
        self.species_colors = {'PICOL': '#e41a1c',
                                'PIFL2': '#377eb8',
                                'ABLAL': '#4daf4a',
                                'PIEN': '#984ea3'}
        

        self.orig_files = [os.path.join(kwargs['orig'], file) for file in os.listdir(kwargs['orig']) if ".h5" in file]
        self.pca_files = [os.path.join(kwargs['pca_dir'], file) for file in os.listdir(kwargs['pca_dir']) if ".npy" in file]
        self.tree_tops_files = [os.path.join(tree_tops_dir, file) for file in os.listdir(tree_tops_dir) if ".geojson" in file]
        self.rgb_files = [os.path.join(rgb_dir, file) for file in os.listdir(rgb_dir) if ".tif" in file]
        
        def make_dict(file_list, param_1, param_2):
            return {f"{file.split('_')[param_1]}_{file.split('_')[param_2]}": file for file in file_list}

        self.orig_dict = make_dict(self.orig_files, -3, -2)
        self.pca_dict = make_dict(self.pca_files, -4, -3)
        self.tree_tops_dict = make_dict(self.tree_tops_files, -3, -2)
        self.rgb_dict = make_dict(self.rgb_files, -3, -2)

        
        self.rng = np.random.default_rng(42)
        self.taxa = {key: ix for ix, key in enumerate(self.data_gdf['taxonID'].unique())}
        
        if use_tt:
            if not os.path.exists(data_gdf):
                self.data_gdf = self.pick_ttops()
                with open(data_gdf, 'wb') as f:
                    pickle.dump(self.data_gdf, f)
            else:
                with open(data_gdf, 'rb') as f:
                    self.data_gdf = pickle.load(f)
        

        # self.data_gdf = gpd.GeoDataFrame(self.data_gdf, geometry=gpd.points_from_xy(self.data_gdf.easting_tree, self.data_gdf.northing_tree), crs='EPSG:32613')
        # self.data_gdf.geometry = self.data_gdf.buffer(self.data_gdf['maxCrownDiameter']*0.9/2)

       

        if object_split:
            self.train, self.valid, self.test = self.get_splits(train_split, valid_split, test_split)
        else:
            self.train_plots, self.valid_plots, self.test_plots = self.split_plots().values()
            self.train = self.data_gdf.loc[self.data_gdf.plotID.isin(list(self.train_plots.index))]
            self.valid = self.data_gdf.loc[self.data_gdf.plotID.isin(list(self.valid_plots.index))]
            self.test = self.data_gdf.loc[self.data_gdf.plotID.isin(list(self.test_plots.index))]
        self.class_weights = cw.compute_class_weight(class_weight='balanced', classes=self.data_gdf['taxonID'].unique(), y=self.train['taxonID'])


    def make_spectrographs(self, filters={}, scale='Tree', save_dir=None):
        grouped_files = self.data_gdf.groupby(['plotID'])
        grouped_files.apply(self._per_plot_spectro, self, filters, scale, save_dir)

    @staticmethod
    def _per_plot_spectro(df, vd, filters, scale, save_dir):

        # Takes a dataframe with a single plot id and plots the spectra of all contained trees

        #Holders for legend handles and labels, lets us make a cleaner legend
        handles = []
        labels = []
        
        key = df.iloc[0]['file_coords']

        west = int(key.split('_')[0])
        north = int(key.split('_')[1]) + 1000
        meter_affine = from_origin(west, north, 1, 1)
        decimeter_affine = from_origin(west, north, .1, .1)

        scene = hp.pre_processing(os.path.join(vd.orig_dict[key]), get_all=True)

        tru_color = hp.pre_processing(os.path.join(vd.orig_dict[key]), utils.get_viz_bands())
        tru_color = utils.make_rgb(tru_color['bands'])
        tru_color = exposure.adjust_gamma(tru_color, 0.5)

        rgb = rs.open(vd.rgb_dict[key])
        big_img = rgb.read()
        big_img = rearrange(big_img, 'c h w -> h w c')/255.0
              
        row = df.iloc[0]
        x_min = row['easting_plot'] - (row['plotSize']**(1/2)/2)
        x_max = row['easting_plot'] + (row['plotSize']**(1/2)/2)

        y_min = row['northing_plot'] - (row['plotSize']**(1/2)/2)
        y_max = row['northing_plot'] + (row['plotSize']**(1/2)/2)
        

        x = [x_min, x_max, x_max, x_min, x_min]
        y = [y_max, y_max, y_min, y_min, y_max]
        points = list(zip(x, y))
        s = gpd.GeoSeries([Polygon(points)], crs='EPSG:32613')


        open_ttops = gpd.read_file(vd.tree_tops_dict[key])

        open_ttops = open_ttops.clip(s)
        

        x_min, x_max, y_min, y_max = row.x_min, row.x_max, row.y_min, row.y_max

        xs = [r.geometry.x - west - x_min for _,r in open_ttops.iterrows()]
        ys = [(1000 - (r.geometry.y - (north - 1000)))-y_min for _,r in open_ttops.iterrows()]

        big_xs = [(r.geometry.x - west - x_min)*10 for _,r in open_ttops.iterrows()]
        big_ys = [((1000 - (r.geometry.y - (north - 1000)))-y_min)*10 for _,r in open_ttops.iterrows()]
        big_ticks = np.arange(5, 400, 10)

        tc = tru_color[y_min:y_max,x_min:x_max,...]

        big_rgb = big_img[y_min*10:y_max*10,x_min*10:x_max*10,...]

        bands = scene['meta']['spectral_bands'][5:-5]
        loaded_img = vd.process_hs_errors(scene["bands"][:,:,5:-5])

        plotID = df.iloc[0]['plotID']

        if 'ndvi' in filters:
            ndvi = hp.pre_processing(os.path.join(vd.orig_dict[key]), utils.get_bareness_bands())["bands"]
            
            ndvi = utils.get_ndvi(ndvi)
       
            ndvi_mask = ndvi < filters['ndvi']
            loaded_img[ndvi_mask] = np.nan
        
        if 'shadow' in filters:
            shadow = hp.pre_processing(os.path.join(vd.orig_dict[key]), utils.get_shadow_bands())["bands"]
            
            shadow = utils.han_2018(shadow)
            shadow_mask = shadow < filters['shadow']
            loaded_img[shadow_mask] = np.nan

        if scale == 'Plot':
            fig, ax = plt.subplots()

        for ix, row in df.iterrows():
            tree_poly = row.geometry
            tree_id = vd.species_lookup[row.taxonID]
            tree_color = vd.species_colors[row.taxonID]
            
            
            big_tree_alpha, big_tree_mask = make_tree_mask(tree_poly, decimeter_affine, (y_min,y_max,x_min,x_max), 10, 1.0, 0.5, all_touched=False)
            tree_alpha, tree_mask = make_tree_mask(tree_poly, meter_affine, (y_min,y_max,x_min,x_max), 1, 2.5, 1.0, all_touched=False)

            tree_data = loaded_img[tree_mask]
            if scale == 'Tree':
                fig, ax = plt.subplots(2, 2, figsize=(10,10))
                ax=np.ravel(ax)
                for pix in tree_data:
                    ax[0].plot(bands, pix)
                    per_change = np.diff(pix) / pix[1:]
                    ax[1].plot(bands[1:], per_change)
                    ax[1].set_ylim((-1,1))
                    
                ax[2].imshow(tc*tree_alpha)
                ax[2].scatter(xs, ys)
                ax[3].imshow(big_rgb*big_tree_alpha)
                ax[3].scatter(big_xs, big_ys)
                ax[3].set_xticks(big_ticks)
                ax[3].set_yticks(big_ticks)
                ax[3].grid()
                plt.suptitle(tree_id + ' - ' + plotID)
                if save_dir is not None:
                    plt.savefig(os.path.join(save_dir, f'{plotID}_{tree_id}{ix}.png'))
                plt.tight_layout()
                plt.show()

            if scale == 'Plot':

                pixel_mean = np.mean(tree_data, axis=0)

                per_change = np.diff(pixel_mean) / pixel_mean[1:]
                #ax.plot(bands[1:], per_change, label=tree_id, color=tree_color)
                ax.plot(bands, pixel_mean, label=tree_id, color=tree_color)

                cur_handles, cur_labels = ax.get_legend_handles_labels()
                for h, l in zip(cur_handles, cur_labels):
                    if l not in labels:
                        labels.append(l)
                        handles.append(h)


        if scale == 'Plot':
        #ax.set_ylim(-1, 1)
            plt.legend(handles, labels)
            if save_dir is not None:
                    plt.savefig(os.path.join(save_dir, f'{plotID}.png'))
            plt.show()


        return None

    
    @staticmethod
    def process_hs_errors(hs_file):
        bad_mask = np.zeros((1000,1000), dtype=bool)
        for i in range(0, hs_file.shape[-1]):
            z = hs_file[:,:,i]
            y = z>1
            bad_mask += y
        hs_file[bad_mask] = np.nan

        return hs_file



    # def make_spectrographs(self, title, gdf_type):
    #     taxa = self.taxa.keys()
    #     stored_stats = {t: [np.zeros((416,), dtype=np.float32), 0] for t in taxa}
    #     for k, v in self.valid_files.items():
           
    #         scene = hp.pre_processing(self.orig_dict[k], get_all=True)
    #         bands = scene['meta']['spectral_bands'][5:-5]
    #         scene = scene["bands"][:,:,5:-5]
    #         if self.ndvi_filter is not None:
    #             ndvi = hp.pre_processing(os.path.join(self.orig_dict[k]), utils.get_bareness_bands())["bands"]
    #             ndvi = utils.get_ndvi(ndvi)
    #             ndvi_mask = ndvi > self.ndvi_filter

    #         west = int(k.split('_')[0])
    #         north = int(k.split('_')[1]) + 1000
    #         affine = from_origin(west, north, 1, 1)

    #         for taxon in taxa:
    #             if gdf_type == 'all':
    #                 tmp_gdf = self.data_gdf.loc[(self.data_gdf['taxonID'] == taxon) & (self.data_gdf['file_coords'] == k)]
    #             if gdf_type == 'train':
    #                 tmp_gdf = self.train.loc[(self.data_gdf['taxonID'] == taxon) & (self.data_gdf['file_coords'] == k)]
    #             if len(tmp_gdf) > 0:
                    
    #                 mask = rf.geometry_mask(tmp_gdf.geometry, (1000,1000), affine, invert=True)
    #                 if self.ndvi_filter is not None:
    #                     mask = mask + ndvi_mask
    #                 selected_data = scene[mask]
    #                 to_add = np.nansum(selected_data, axis=0)
    #                 stored_stats[taxon][0] += to_add
    #                 stored_stats[taxon][1] += selected_data.shape[0]
    #                 #print('here')
    #     calced_stats = {}
    #     for k, v in stored_stats.items():
    #         sums = v[0]
    #         count = v[1]

    #         mean = sums/count
    #         calced_stats[k] = mean
    #         plt.plot(bands, mean, label=self.species_lookup[k])
    #         #plt.title(k)
    #     class_weights_list = []
    #     for k, v in stored_stats.items():
    #         to_add = [k] * v[1]
    #         class_weights_list = class_weights_list + to_add
    #     self.class_weights = cw.compute_class_weight(class_weight='balanced', classes=self.data_gdf['taxonID'].unique(), y=class_weights_list)
    #     print(self.class_weights)
    #     plt.legend()
    #     plt.title(title)
    #     plt.xlabel('Wavelength (nm)', fontsize='large')
    #     plt.ylabel('Mean Reflectance', fontsize='large')
    #     plt.show()
    #     # print('here')


    def save_orig_to_geotiff(self, key, save_loc, thresh=None, mode='rgb'):
        img = self.orig_dict[key]

        if mode == 'rgb':
            rgb = hp.pre_processing(img, wavelength_ranges=utils.get_viz_bands())
            rgb = hp.make_rgb(rgb["bands"])
            rgb = exposure.adjust_gamma(rgb, 0.5)
            scene = rearrange(rgb, 'h w c -> c h w')

        if mode == 'mpsi':
            mpsi = hp.pre_processing(os.path.join(self.orig_dict[key]), utils.get_shadow_bands())["bands"]
            mpsi = utils.han_2018(mpsi)
            if thresh is not None:
                thresh_mask = mpsi>thresh
                mpsi[thresh_mask] = 1
                mpsi[~thresh_mask] = 0
            scene = mpsi[np.newaxis, ...]

        if mode == 'ndvi':
            ndvi = hp.pre_processing(os.path.join(self.orig_dict[key]), utils.get_bareness_bands())["bands"]
                            
            ndvi = utils.get_ndvi(ndvi)
            if thresh is not None:
                thresh_mask = ndvi>thresh
                ndvi[thresh_mask] = 1
                ndvi[~thresh_mask] = 0
            scene = ndvi[np.newaxis, ...]

        west = int(key.split('_')[0])
        north = int(key.split('_')[1]) + 1000
        affine = from_origin(west, north, 1, 1)

        new_img = rs.open(
            save_loc,
            'w',
            driver="GTiff",
            height=1000,
            width=1000,
            count=1,
            dtype=scene.dtype,
            crs='+proj=utm +zone=13 +datum=WGS84 +units=m +no_defs +type=crs',
            transform=affine
        )

        new_img.write(scene)
        new_img.close()

    def save_pca_to_geotiff(self, key, save_loc):
        img = self.pca_dict[key]
        pca = np.load(img).astype(np.float32)
        pca = rearrange(pca, 'h w c -> c h w')

        west = int(key.split('_')[0])
        north = int(key.split('_')[1]) + 1000
        affine = from_origin(west, north, 1, 1)

        new_img = rs.open(
            save_loc,
            'w',
            driver="GTiff",
            height=1000,
            width=1000,
            count=20,
            dtype=pca.dtype,
            crs='+proj=utm +zone=13 +datum=WGS84 +units=m +no_defs +type=crs',
            transform=affine
        )

        pca[pca!=pca] = 0
        new_img.write(pca)
        new_img.close()
    
    def save_plot_outlines(self, save_loc, df):
        grouped_files = df.groupby(['plotID'])
        to_save = grouped_files.apply(self._make_plot_outlines)
        to_save = to_save.reset_index()
        to_save = to_save.rename(mapper={0: 'geometry'}, axis=1)
        to_save = to_save.set_crs(crs='EPSG:32613')
        to_save.to_file(save_loc)
    
    @staticmethod
    def _make_plot_outlines(df):
        row = df.iloc[0]
        x_min = row['easting_plot'] - (row['plotSize']**(1/2)/2)
        x_max = row['easting_plot'] + (row['plotSize']**(1/2)/2)

        y_min = row['northing_plot'] - (row['plotSize']**(1/2)/2)
        y_max = row['northing_plot'] + (row['plotSize']**(1/2)/2)
        x = [x_min, x_max, x_max, x_min, x_min]
        y = [y_max, y_max, y_min, y_min, y_max]
        points = list(zip(x, y))
        s = gpd.GeoSeries([Polygon(points)], crs='EPSG:32613')
        to_return = gpd.GeoDataFrame(s)
        return to_return

        

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
        if valid_prop != 0:
            train, t_v = ms.train_test_split(self.data_gdf, test_size=(valid_prop+test_prop), train_size=train_prop, random_state=42, stratify=self.data_gdf['taxonID'])
            test, valid = ms.train_test_split(t_v, test_size=valid_prop/(valid_prop+test_prop), train_size=test_prop/(valid_prop+test_prop), random_state=42, stratify=t_v['taxonID'])
            return train, valid, test
        else:
            train, test = ms.train_test_split(self.data_gdf, test_size=test_prop, train_size=train_prop, random_state=42, stratify=self.data_gdf['taxonID'])
            return train, None, test

    def render_valid_patch(self, save_dir, split, out_size=20, multi_crop=1, num_channels=16, key_label='pca', filters=[]):
        if split == 'train':
            data = self.train
        if split == 'valid':
            data = self.valid
        if split == 'test':
            data = self.test
        if split == 'all':
            data = self.data_gdf

        loaded_key = None
        ndvi_key = None
        shadow_key = None
        rgb_key = None
        data = data.sort_values('file_coords')
        for ix, row in data.iterrows():
            key = row['file_coords']
            if key != loaded_key:
                if key_label != 'hs':
                    pca = np.load(self.pca_dict[key]).astype(np.float32)
                if key_label == 'hs':
                    pca = hp.pre_processing(os.path.join(self.orig_dict[key]), get_all=True)["bands"][:,:,5:-5]
                    bad_mask = np.zeros((1000,1000), dtype=bool)
                    for i in range(0, pca.shape[-1]):
                        z = pca[:,:,i]
                        y = z>1
                        bad_mask += y
                    pca[bad_mask] = np.nan
                loaded_key = key

            west = int(key.split('_')[0])
            north = int(key.split('_')[1]) + 1000
            affine = from_origin(west, north, 1, 1)
            
            
            #rad = row['maxCrownDiameter']*self.crown_diameter_mult/2
            rad=3
            target_size = 7

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

                       
                if pca_crop.shape == (out_size, out_size, num_channels):
                    pca_crop = torch.tensor(pca_crop)
                    pca_crop = rearrange(pca_crop, 'h w c -> c h w')
                    masks = {}
                    if 'ndvi' in filters:
                        if key != ndvi_key:
                            ndvi = hp.pre_processing(os.path.join(self.orig_dict[key]), utils.get_bareness_bands())["bands"]
                            
                            ndvi = utils.get_ndvi(ndvi)
                            ndvi_key = key
                        ndvi_clip = ndvi[y_min:y_max, x_min:x_max]
                        #ndvi_mask = ndvi < self.ndvi_filter
                        masks['ndvi'] = ndvi_clip
                    if 'shadow' in filters:
                        if key != shadow_key:
                            shadow = hp.pre_processing(os.path.join(self.orig_dict[key]), utils.get_shadow_bands())["bands"]
                            
                            shadow = utils.han_2018(shadow)
                            shadow_key = key
                        shadow_clip = shadow[y_min:y_max, x_min:x_max]
                        #shadow_mask = shadow < 0.03
                        masks['shadow'] = shadow_clip
                    
                    if 'rgb_z_max' in filters:
                        if key != rgb_key:
                            rgb = rs.open(self.rgb_dict[key])
                            img = rgb.read()

                            
                            stats = torch.load(self.stats_loc)
                            
                            scaler = StandardScaler()
                            cur_stats = stats['rgb']
                            scaler.scale_ = cur_stats['scale']
                            scaler.mean_ = cur_stats['mean']
                            scaler.var_ = cur_stats['var']

                            img = rearrange(img, 'c h w -> (h w) c')
                            img = scaler.transform(img)
                            img = rearrange(img, '(h w) c -> c h w', h=10000, w=10000)
                            img = torch.from_numpy(img).float()
                            
                            #img = tf.center_crop(img, [512,512])
                            img = torch.nn.functional.interpolate(img.unsqueeze(0), scale_factor=0.1).squeeze()

                            img = torch.argmax(img, dim=0)
                            rgb_key = key
                        clip_img = img[y_min:y_max, x_min:x_max]
                        green_mask = clip_img == 1
                        masks['rgb_z_max'] = green_mask
                    #pca_crop[mask] = 0
                    

                    label = torch.zeros((1000, 1000), dtype=torch.float32)
                    #label[self.taxa[taxa]] = 1.0
                    label = label - 1
                    target_mask = rf.geometry_mask([row.geometry], (1000,1000), affine, invert=True, all_touched=True)
                    label[target_mask] = self.taxa[taxa]
                    label = label[y_min:y_max, x_min:x_max]

                    #label[neg_y_pad:neg_y_pad+target_size+1, neg_x_pad:neg_x_pad+target_size+1] = self.taxa[taxa]

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
        if self.crown_diameter_mult is not None:
           data_gdf.geometry = data_gdf.buffer(data_gdf['maxCrownDiameter']*self.crown_diameter_mult/2)
        else:
            data_gdf.geometry = data_gdf.buffer(self.constant_diameter)
        

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

    
    def split_plots(self, splits={'train': 0.7, 'valid':0.0, 'test':0.3}):
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
                #area= cur.area.sum()
                #pc[df.iloc[0]['plotID']][tx] = area
                pc[df.iloc[0]['plotID']][tx] = len(cur)




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
        row = df.iloc[0]
        coords = row['file_coords']

        x_min = row['easting_plot'] - (row['plotSize']**(1/2)/2)
        x_max = row['easting_plot'] + (row['plotSize']**(1/2)/2)

        y_min = row['northing_plot'] - (row['plotSize']**(1/2)/2)
        y_max = row['northing_plot'] + (row['plotSize']**(1/2)/2)
        x = [x_min, x_max, x_max, x_min, x_min]
        y = [y_max, y_max, y_min, y_min, y_max]
        points = list(zip(x, y))
        s = gpd.GeoSeries([Polygon(points)], crs='EPSG:32613')


        open_ttops = gpd.read_file(vd.tree_tops_dict[coords])

        open_ttops = open_ttops.clip(s)

        df['cur_buffer'] = df.buffer(vd.ttops_search_buffer)

        clip = open_ttops.clip(df.cur_buffer)


        #TODO: Switch to spatial index

        def find_match(row, clip):
            distances = clip.distance(row.geometry)
            distances = distances.loc[distances<vd.ttops_search_buffer]
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

        if vd.crown_diameter_mult is not None:
            df.geometry = df.buffer(df['maxCrownDiameter']*vd.crown_diameter_mult/2)
        else:
            df.geometry = df.buffer(vd.constant_diameter)
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

                        all = all.sort_values('height', ascending=False)
                        top_taxon = all.iloc[0].taxonID
                        to_remove = all.iloc[1:]
                        to_de_intersect = to_de_intersect + list(to_remove.index)

            df = df.drop(to_de_intersect)

            # fig, ax = plt.subplots(1,1)
            # df.plot(column='taxa_key', ax=ax, legend=True)
            # plt.show()


        return df
   

    def pick_ttops(self):
        grouped_files = self.data_gdf.groupby('plotID')
        selected = grouped_files.apply(self._select_ttops, self)
        selected = selected.drop('plotID', axis=1).reset_index()
        return selected




    
    def map_plots(self, save_dir):
        grouped_files = self.data_gdf.groupby(['file_coords', 'plotID'])
        grouped_files.apply(self._map_plot, self.orig_dir, save_dir, self.site_name, self.site_prefix)



##HELPER FUNCTIONS



def make_tree_mask(polygon, transform, slice_params, scale, upper, lower, all_touched=False):
    mask = rf.geometry_mask([polygon], [1000*scale, 1000*scale], transform=transform, all_touched=all_touched, invert=True)
    ym, yma, xm, xma = slice_params
    cropped_mask = mask[ym*scale:yma*scale, xm*scale:xma*scale, ...]
    cropped_mask = cropped_mask * upper
    cropped_mask[cropped_mask == 0] = lower
    return cropped_mask[...,np.newaxis], mask
        

    





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
    RGB_DIR = 'C:/Users/tonyt/Documents/Research/datasets/rgb/NEON.D13.NIWO.DP3.30010.001.2020-08.basic.20220814T183511Z.RELEASE-2022/'

    valid = Validator(file=VALID_FILE, 
                    pca_dir=PCA_DIR, 
                    site_name='NIWO',
                    train_split=0.7,
                    test_split=0.2, 
                    valid_split=0.1,
                    num_classes=NUM_CLASSES, 
                    plot_file=PLOT_FILE, 
                    tree_tops_dir=TREE_TOPS_DIR,
                    curated=CURATED_FILE, 
                    orig=ORIG_DIR, 
                    rgb_dir=RGB_DIR,
                    stats_loc='C:/Users/tonyt/Documents/Research/datasets/tensors/rgb_blocks/stats/stats.npy',
                    prefix='D13',
                    use_tt=True,
                    scholl_filter=False,
                    scholl_output=False,
                    filter_species = 'SALIX',
                    object_split=True,
                    data_gdf='3m_search_0.5_crowns_ttops_clipped_to_plot.pkl.pkl',
                    crown_diameter_mult=0.5,
                    constant_diameter=1.5,
                    ttops_search_buffer=3,
                    ndvi_filter=0.5)

    #valid.make_spectrographs(filters={'ndvi': 0.2, 'shadow': 0.03})
    valid.make_spectrographs()

    #valid.save_orig_to_geotiff('451000_4432000', 'C:/Users/tonyt/Documents/Research/rendered_imgs/451000_4432000_mpsi_threshed_0.03.tif', thresh=0.03, mode='mpsi')

    # valid.render_valid_patch('C:/Users/tonyt/Documents/Research/datasets/tensors/rf_test/rgb_mask_plot/test', 'test', out_size=20, num_channels=16, key_label='pca', filters=['ndvi', 'shadow'])
    # valid.render_valid_patch('C:/Users/tonyt/Documents/Research/datasets/tensors/rf_test/rgb_mask_plot/train', 'train', out_size=20, num_channels=16, key_label='pca', filters=['ndvi', 'shadow'])


    print(valid.taxa)
    print(valid.class_weights)




