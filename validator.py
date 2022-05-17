import wave
from matplotlib.widgets import Slider
from matplotlib import colors
from sklearn import cluster
import geopandas as gpd
from rasterio.plot import show
import rasterio as rs
from rasterio.transform import from_origin
import rasterio.features as rf
from rasterio.crs import CRS
import torch
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import h5_helper as hp
import inference
import os
import models
import utils
from skimage import exposure
from scipy.stats import linregress
from einops import rearrange
from sklearn.decomposition import PCA, IncrementalPCA
import torchvision.transforms as tt

#TODO: Clean this mess up
#Switch plots to plotly? Good for overlays
class Validator():
    def __init__(self, struct=False, **kwargs):
        self.file = kwargs["file"]
        self.img_dir = kwargs["img_dir"]
        self.struct = struct
        self.curated = kwargs['curated']
        
        self.num_clusters = kwargs["num_clusters"]
        self.site_name = kwargs["site_name"]
        self.plot_file = kwargs['plot_file']
        self.valid_data = self.get_plot_data()
        self.valid_files = self.get_valid_files()

        def make_dict(file_list, param_1, param_2):
            return {f"{file.split('_')[param_1]}_{file.split('_')[param_2]}": file for file in file_list}

        if struct:
            self.chm_files = [os.path.join(kwargs['chm'], file) for file in os.listdir(kwargs['chm']) if ".tif" in file]
            self.azm_files = [os.path.join(kwargs['azm'], file) for file in os.listdir(kwargs['azm']) if ".npy" in file]

            self.chm_dict = make_dict(self.chm_files, -3, -2)
            self.azm_dict = make_dict(self.azm_files, -4, -3)

        self.cluster_groups = set()

        self.last_cluster = {}

    def get_plot_data(self):
        #data = pd.read_csv(self.file, usecols=['siteID', 'plotID', 'plantStatus', 'ninetyCrownDiameter', 'canopyPosition', 'taxonID', 'ninetyCrownDiameter'])
        data = pd.read_csv(self.file, usecols=['siteID', 'plotID', 'plantStatus', 'taxonID', 'ninetyCrownDiameter', 'canopyPosition', 'easting', 'northing', 'height'])
        data = data.loc[data['siteID'] == self.site_name]
        curated = pd.read_csv(self.curated)
        curated = curated.rename(columns={'adjEasting': 'easting',
                                          'adjNorthing': 'northing'})
        data_with_coords = curated.loc[(curated.easting == curated.easting) & (curated.northing==curated.northing)]
       
        data = data.loc[(data['plantStatus'] != 'Dead, broken bole') & (data['plantStatus'] != 'Downed') & (data['plantStatus'] != 'No longer qualifies') & (data['plantStatus'] != 'Lost, fate unknown') & (data['plantStatus'] != 'Removed') & (data['plantStatus'] != 'Lost, presumed dead')]
        #data_with_coords = data.loc[(data.easting == data.easting) & (data.height == data.height) & (data.ninetyCrownDiameter == data.ninetyCrownDiameter)]
        data_gdf = gpd.GeoDataFrame(data_with_coords, geometry=gpd.points_from_xy(data_with_coords.easting, data_with_coords.northing), crs='EPSG:32618')
        #data_gdf['crowns'] = data_gdf.geometry.buffer(data_gdf['ninetyCrownDiameter']/2)
        data_gdf['crowns'] = data_gdf.geometry.buffer(5)


        test_gdf = data_gdf.loc[(data_gdf.easting > 731000) & (data_gdf.easting < 732000) & (data_gdf.northing > 4713000) & (data_gdf.northing < 4714000)]
        #test_gdf = data_gdf.loc[data_gdf.plotID == 'HARV_034']
        img_loc = 'W:/Classes/Research/datasets/hs/original/NEON.D01.HARV.DP3.30006.001.2019-08.basic.20220501T135554Z.RELEASE-2022/NEON_D01_HARV_DP3_731000_4713000_reflectance.h5'
        test = hp.pre_processing(img_loc, wavelength_ranges=utils.get_viz_bands())
        test_rgb = hp.make_rgb(test['bands'])
        test_rgb = rearrange(test_rgb, 'h w c -> c h w')
        transform = from_origin(731000, 4714000, 1, 1)
        test_rgb = exposure.adjust_gamma(test_rgb, 0.5)
        #crs_code = CRS.from_string(test['meta']['proj'])
        #need to replace UTM with utm in proj string
        rs_test = rs.open('test1.tif', 'w+', driver='GTiff',
                            height = test_rgb.shape[1], width = test_rgb.shape[2], 
                            count=3, dtype=np.float32,
                            crs='+proj=utm +zone=18 +ellps=WGS84 +datum=WGS84 +units=m +no_defs',
                            transform=transform)
        rs_test.write(test_rgb)

        tree_outlines = rf.geometry_mask(test_gdf.crowns, (1000,1000), transform=rs_test.transform, invert=True, all_touched=True)
        trees = np.ma.masked_where(tree_outlines == False, tree_outlines)
        cmap = colors.ListedColormap(['red'])
        fig, ax = plt.subplots(figsize=(20,20))
        ax.imshow(rearrange(rs_test.read(), 'c h w -> h w c'))
        im = ax.imshow(trees, cmap=cmap, alpha=0.2)
        slider = self._make_slider(fig, im)
        plt.show()


        data['approx_sq_m'] = ((data['ninetyCrownDiameter']/2)**2) * np.pi

        
        plots = pd.read_csv(self.plot_file, usecols=['plotID', 'siteID', 'subtype', 'easting', 'northing', 'plotSize'])
        plots = plots.loc[plots['siteID'] == self.site_name]
        plots = plots.loc[plots['subtype'] == 'basePlot']

        data = data.merge(plots, how='left', on='plotID')

        data = data.rename(columns={
                                    'easting_x': 'easting_tree',
                                    'northing_x': 'northing_tree',
                                    'easting_y': 'easting_plot',
                                    'northing_y': 'northing_plot'
        })

        data["file_west_bound"] = data["easting_plot"] - data["easting_plot"] % 1000
        data["file_south_bound"] = data["northing_plot"] - data["northing_plot"] % 1000

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
        data['sun'] = data['canopyPosition'].str.contains('sun')

        return data


    def get_valid_files(self):
        if self.img_dir is not None:
            coords = list(self.valid_data['file_coords'].unique())

            all_files = os.listdir(self.img_dir)
            valid_files = {coord:os.path.join(self.img_dir,file) for file in all_files for coord in coords if coord in file}
            return valid_files
        else:
            return None

    @staticmethod
    def _extract_plot(df, img_dir, save_dir):
        first_row = df.iloc[0]
        coords = first_row['file_coords']
        open_file = os.path.join(img_dir, f'NEON_D01_HARV_DP3_{coords}_reflectance.h5')
        all_bands = hp.pre_processing(open_file, get_all=True)
        all_bands['bands'] = all_bands['bands'][first_row['y_min']:first_row['y_max'], first_row['x_min']:first_row['x_max'], :]
        all_bands['meta']['plotID'] = first_row['plotID']
        all_bands['meta']['original_file'] = f'NEON_D01_HARV_DP3_{coords}_reflectance.h5'
        #save plot by name, centroid, and original file
        save_name = f'plot_subset_{first_row["plotID"]}_eastingcentroid_{int(first_row["easting"])}_northingcentroid_{int(first_row["northing"])}_fromfile_{coords}.pk'
        #np.save(os.path.join(save_dir, save_name), all_bands)
        with open(os.path.join(save_dir, save_name), 'wb') as f:
            pickle.dump(all_bands, f)

    def extract_plots(self, save_dir):
        grouped_files = self.valid_data.groupby(['file_coords', 'plotID'])
        grouped_files.apply(self._extract_plot, self.img_dir, save_dir)

    @staticmethod
    def _map_plot(df, img_dir, save_dir):
        fig, ax = plt.subplots(figsize=(10, 10))
        row = df.iloc[0]
        coords = row['file_coords']
        open_file = os.path.join(img_dir, f'NEON_D01_HARV_DP3_{coords}_reflectance.h5')
        rgb = hp.pre_processing(open_file, wavelength_ranges=utils.get_viz_bands())
        rgb = hp.make_rgb(rgb["bands"])
        rgb = exposure.adjust_gamma(rgb, 0.5)
        ax.imshow(rgb)
        x = [row['x_min'], row['x_max'], row['x_max'], row['x_min'], row['x_min']]
        y = [row['y_max'], row['y_max'], row['y_min'], row['y_min'], row['y_max']]
        ax.plot(x, y)
        ax.set_title(f'Original File: NEON_D01_HARV_DP3_{coords}_reflectance.h5 \n PlotID: {row["plotID"]}')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'NEON_D01_HARV_DP3_{coords}_Plot_{row["plotID"]}.png'))
        plt.close()

    
    @staticmethod
    def _map_trees(df, img_dir, save_dir):
        fig, ax = plt.subplots(figsize=(10, 10))
        row = df.iloc[0]
        coords = row['file_coords']
        open_file = os.path.join(img_dir, f'NEON_D01_HARV_DP3_{coords}_reflectance.h5')
        rgb = hp.pre_processing(open_file, wavelength_ranges=utils.get_viz_bands())
        rgb = hp.make_rgb(rgb["bands"])
        rgb = exposure.adjust_gamma(rgb, 0.5)

        rgb = rgb[row['y_min']:row['y_max'], row['x_min']:row['x_max']]
        plt.imshow(rgb)
        df['tree_x'] = df['tree_x'] - df['x_min']
        df['tree_y'] = df['tree_y'] - df['y_min']

        trees_df = df.loc[(df['tree_x'] == df['tree_x'])]
        for ix, row in trees_df.iterrows():
            ax.plot(row['tree_x'], row['tree_y'], marker="o")
            ax.annotate(row['taxonID'], (row['tree_x'], row['tree_y']), textcoords='offset points', xytext= (0, 5), ha='center', color='white', fontsize='xx-large')
        fig.tight_layout()
        plt.savefig(os.path.join(save_dir, f'{row["plotID"]}_{coords}_trees.png'))
        plt.close()



    def map_trees(self, save_dir):
        grouped_files = self.valid_data.groupby(['file_coords', 'plotID'])
        grouped_files.apply(self._map_trees, self.img_dir, save_dir)

    def do_plot_inference(self, save_dir, model):
        grouped_files = self.valid_data.groupby(['file_coords', 'plotID'])
        grouped_files.apply(self._plot_inference, self, save_dir, model)

    @staticmethod
    def _plot_inference(df, validator, save_dir, model):
        row = df.iloc[0]
        file_key = row['file_coords']
        plot_id = row['plotID']
        if file_key not in validator.last_cluster:
            f = validator.valid_files[file_key]
            if not validator.struct:
                clustered = inference.swav_inference_big(model, f)
            else:
                c = validator.chm_dict[file_key]
                a = validator.azm_dict[file_key]
                clustered = inference.swav_inference_big_struct_4(model, f, c, a)
            validator.last_cluster[file_key] = clustered
        else:
            clustered = validator.last_cluster[file_key]
        plt.imsave(os.path.join(save_dir, f"{plot_id}_viz_full.png"),clustered, cmap='tab20')
       
        clustered = clustered[row['y_min']:row['y_max'], row['x_min']:row['x_max']]
        plt.imsave(os.path.join(save_dir, f"{plot_id}_viz.png"),clustered, cmap='tab20')
        plt.bar(*np.unique(clustered, return_counts=True))
        
        plt.title(plot_id)
        plt.xlabel('Classification')
        plt.ylabel('Pixel Count')
        #plt.xlim(0, 10)

        plt.savefig(os.path.join(save_dir, plot_id + "_hist.png"))
        plt.close()


        #plt.imsave(os.path.join(save_dir, f"{plot_id}_viz.png"),clustered, cmap='tab20')

    
    def map_plots(self, save_dir):
        grouped_files = self.valid_data.groupby(['file_coords', 'plotID'])
        grouped_files.apply(self._map_plot, self.img_dir, save_dir)

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
        # for ix, row in data.iterrows():
        #     # x = [row['x_min'], row['x_max'], row['x_max'], row['x_min'], row['x_min']]
        #     # y = [row['y_max'], row['y_max'], row['y_min'], row['y_min'], row['y_max']]
        #     x= (row['x_min'] + row['x_max'])//2
        #     y= (row['y_min'] + row['y_max'])//2
        #     ax.plot(x, y, marker="o")
        #     ax.annotate(row['taxonID'], (x, y), textcoords='offset points', xytext= (0, 5), ha='center')
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
        # df = pd.DataFrame.from_dict(self.cluster_dict, orient='index', dtype=int)
        # df["sum"] = df.sum(axis=1)
        # sum_row = pd.DataFrame(df.sum(axis=0)).transpose()
        # sum_row = sum_row.rename(index={0:'sum'})
        # df = pd.concat([df, sum_row])
        # return df

    def kappa(self):
        return None
    

def check_predictions(validator: Validator, model, coords, h5_file, save_dir, **kwargs):
    y = inference.do_inference(model, h5_file, True, True, **kwargs)
    np.save(os.path.join(save_dir,coords+".npy"), y)
    validator.validate(coords, y)
    return None

def check_all(validator: Validator, model, save_dir, **kwargs):
    for coord, file in validator.valid_files.items():
        check_predictions(validator, model, coord, file, save_dir, **kwargs)
    return None

def bulk_validation(ckpts_dir, img_dir, save_dir, valid_file, model_type, **kwargs):
    for ckpt in os.listdir(ckpts_dir):
         if ".ckpt" in ckpt:
             model = inference.load_ckpt(model_type, os.path.join(ckpts_dir, ckpt), **kwargs)
             valid = Validator(file=valid_file, img_dir=img_dir, **kwargs)
             ckpt_name = ckpt.replace(".ckpt", "")
             new_dir = os.path.join(save_dir, ckpt_name)
             if not os.path.isdir(new_dir):
                os.mkdir(new_dir)
             check_all(valid, model, new_dir, n_components=kwargs['num_channels'])
             valid.confusion_matrix.to_csv(os.path.join(save_dir, ckpt_name + "conf_matrix.csv"))
    return True

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

def plot_inference(validator, save_dir, model):
    for key, f in validator.valid_files.items():
        clustered = inference.swav_inference_big(model, f)
        plt.bar(*np.unique(clustered, return_counts=True))
        plot_id = f.split("_")[2]+ " "+ f.split("_")[3]
        plt.title(plot_id)
        plt.xlabel('Classification')
        plt.ylabel('Pixel Count')
        #plt.xlim(0, 10)

        plt.savefig(os.path.join(save_dir, plot_id + ".png"))
        plt.close()


        plt.imsave(os.path.join(save_dir, 'viz',  f"{plot_id}.png"),clustered, cmap='tab20')



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










if __name__ == "__main__":
    NUM_CLUSTERS = 60
    NUM_CHANNELS = 30
    PCA_DIR= 'C:/Users/tonyt/Documents/Research/datasets/pca/harv_2022_10_channels'
    PCA = os.path.join(PCA_DIR, 'NEON_D01_HARV_DP3_736000_4703000_reflectance_pca.npy')
    IMG_DIR = 'W:/Classes/Research/datasets/hs/original/NEON.D01.HARV.DP3.30006.001.2019-08.basic.20220501T135554Z.RELEASE-2022'
    OUT_NAME = "test_inference_ckpt_6.npy"
    IMG= os.path.join(IMG_DIR, 'NEON_D01_HARV_DP3_736000_4703000_reflectance.h5')
   
    PLOT_DIR = "C:/Users/tonyt/Documents/Research/datasets/extracted_plots/harv_2022"
    VALID_FILE = "W:/Classes/Research/neon-allsites-appidv-latest.csv"
    CURATED_FILE = "W:/Classes/Research/neon_mapped_harv.csv"
    PLOT_FILE = 'W:/Classes/Research/All_NEON_TOS_Plots_V8/All_NEON_TOS_Plots_V8/All_NEON_TOS_Plot_Centroids_V8.csv'
    CKPTS_DIR = "ckpts/harv_transformer_fixed_augment"
    PRED_DIR = 'validation/harv_simsiam_transformer_0_1/harv_transformer_60_classes_epoch=25'
    PLOT_SUBSET = os.path.join(PLOT_DIR, 'plot_subset_HARV_024_eastingcentroid_726037_northingcentroid_4704513_fromfile_726000_4704000.pk')
    PLOT_VIZ = 'C:/Users/tonyt/Documents/Research/datasets/extracted_plots/harv_2022/area_plots'
    CHM_DIR = 'C:/Users/tonyt/Documents/Research/datasets/chm/harv_2019/NEON_struct-ecosystem/NEON.D01.HARV.DP3.30015.001.2019-08.basic.20220511T165943Z.RELEASE-2022/'
    AZM_DIR = 'C:/Users/tonyt/Documents/Research/datasets/solar_azimuth/harv_2022/'

    PLOT_PKLS = 'C:/Users/tonyt/Documents/Research/datasets/extracted_plots/harv_2022/plot_pickles'
    PLOT_PCA = 'C:/Users/tonyt/Documents/Research/datasets/extracted_plots/harv_2022/plots_pca'

    SAVE_DIR = 'C:/Users/tonyt/Documents/Research/datasets/extracted_plots/harv_2022/experiments/all_struct'

    valid = Validator(file=VALID_FILE, img_dir=PCA_DIR, site_name='HARV', num_clusters=NUM_CLUSTERS, plot_file=PLOT_FILE, struct=True, azm=AZM_DIR, chm=CHM_DIR, curated=CURATED_FILE)
    #TODO: Automate saving these from training
    # configs = [
    #     {'num_channels': 10,
    #     'num_classes': 12,
    #     'azm': False,
    #     'chm': False,
    #     'patch_size': 4,
    #     'log_every': 10,
    #     'max_epochs': 50,
    #     'num_workers': 4,
    #     'img_size': 40,
    #     'use_queue': True,
    #     'extra_labels': 'no_struct_queue',
    #     'save_dir': 'C:/Users/tonyt/Documents/Research/datasets/extracted_plots/harv_2022/experiments/no_struct_queue',
    #     'ckpt': 'ckpts/exp_1/harv_10_channels_12_classes_swav_structure_patch_size_4_no_struct_queue_epoch=49.ckpt'
    #     }
    # ]

    # configs = [
    #      {'num_channels': 10,
    #      'num_classes': 12,
    #      'azm': True,
    #      'chm': True,
    #      'patch_size': 4,
    #      'log_every': 10,
    #      'max_epochs': 50,
    #      'num_workers': 4,
    #      'img_size': 40,
    #      'use_queue': False,
    #      'same_embed': False,
    #      'azm_concat': True,
    #      'chm_concat': True,
    #      'extra_labels': 'struct_concat',
    #      'save_dir': 'C:/Users/tonyt/Documents/Research/datasets/extracted_plots/harv_2022/experiments/struct_concat',
    #      'ckpt': 'ckpts/exp_2/harv_10_channels_12_classes_swav_structure_patch_size_4_struct_concat_epoch=49.ckpt'
    #      },
    #      {'num_channels': 10,
    #      'num_classes': 12,
    #      'azm': True,
    #      'chm': True,
    #      'patch_size': 4,
    #      'log_every': 10,
    #      'max_epochs': 50,
    #      'num_workers': 4,
    #      'img_size': 40,
    #      'use_queue': True,
    #      'same_embed': False,
    #      'azm_concat': True,
    #      'chm_concat': True,
    #      'extra_labels': 'struct_concat_queue',
    #      'save_dir': 'C:/Users/tonyt/Documents/Research/datasets/extracted_plots/harv_2022/experiments/struct_concat_queue',
    #      'ckpt': 'ckpts/exp_2/harv_10_channels_12_classes_swav_structure_patch_size_4_struct_concat_queue_epoch=49.ckpt'
    #      },
    #      {'num_channels': 10,
    #      'num_classes': 12,
    #      'azm': True,
    #      'chm': True,
    #      'patch_size': 4,
    #      'log_every': 10,
    #      'max_epochs': 50,
    #      'num_workers': 4,
    #      'img_size': 40,
    #      'use_queue': True,
    #      'same_embed': False,
    #      'concat': False,
    #      'queue_chunks': 5,
    #      'extra_labels': 'struct_queue_5_chunks',
    #      'save_dir': 'C:/Users/tonyt/Documents/Research/datasets/extracted_plots/harv_2022/experiments/struct_5_chunk_queue',
    #      'ckpt': 'ckpts/exp_2/harv_10_channels_12_classes_swav_structure_patch_size_4_struct_queue_5_chunks_epoch=49.ckpt'
    #      },
    #      {'num_channels': 10,
    #      'num_classes': 12,
    #      'azm': False,
    #      'chm': False,
    #      'patch_size': 4,
    #      'log_every': 10,
    #      'max_epochs': 50,
    #      'num_workers': 4,
    #      'img_size': 40,
    #      'use_queue': True,
    #      'same_embed': False,
    #      'concat': False,
    #      'queue_chunks': 5,
    #      'extra_labels': 'no_struct_queue_5_chunks',
    #      'save_dir': 'C:/Users/tonyt/Documents/Research/datasets/extracted_plots/harv_2022/experiments/no_struct_5_chunk_queue',
    #      'ckpt': 'ckpts/exp_2/harv_10_channels_12_classes_swav_structure_patch_size_4_no_struct_queue_5_chunks_epoch=49.ckpt'
    #      }
    #     ]

    # configs = [
        # {'num_channels': 10,
        #  'num_classes': 12,
        #  'azm': True,
        #  'chm': True,
        #  'patch_size': 4,
        #  'log_every': 10,
        #  'max_epochs': 50,
        #  'num_workers': 4,
        #  'img_size': 40,
        #  'use_queue': True,
        #  'same_embed': False,
        #  'azm_concat': True,
        #  'chm_concat': True,
        #  'queue_chunks': 5,
        #  'extra_labels': 'all_struct_concat_queue_5_chunks',
        #  'save_dir': 'C:/Users/tonyt/Documents/Research/datasets/extracted_plots/harv_2022/experiments/all_struct_concat_queue_5_chunks',
        #  'ckpt': 'ckpts/exp_3/harv_10_channels_12_classes_swav_structure_patch_size_4_all_struct_concat_queue_5_chunks_epoch=49.ckpt'
        #  },
        #  {'num_channels': 10,
        #  'num_classes': 12,
        #  'azm': True,
        #  'chm': True,
        #  'patch_size': 4,
        #  'log_every': 10,
        #  'max_epochs': 50,
        #  'num_workers': 4,
        #  'img_size': 40,
        #  'use_queue': False,
        #  'same_embed': False,
        #  'azm_concat': False,
        #  'chm_concat': True,
        #  'queue_chunks': 5,
        #  'extra_labels': 'azm_add_chm_concat_no_queue',
        #  'save_dir': 'C:/Users/tonyt/Documents/Research/datasets/extracted_plots/harv_2022/experiments/azm_add_chm_concat_no_queue',
        #  'ckpt': 'ckpts/exp_3/harv_10_channels_12_classes_swav_structure_patch_size_4_azm_add_chm_concat_no_queue_epoch=49.ckpt'
        #  },
    #       {'num_channels': 10,
    #      'num_classes': 12,
    #      'azm': True,
    #      'chm': True,
    #      'patch_size': 4,
    #      'log_every': 10,
    #      'max_epochs': 50,
    #      'num_workers': 4,
    #      'img_size': 40,
    #      'use_queue': False,
    #      'same_embed': False,
    #      'azm_concat': True,
    #      'chm_concat': False,
    #      'queue_chunks': 5,
    #      'extra_labels': 'chm_add_azm_concat_no_queue',
    #       'save_dir': 'C:/Users/tonyt/Documents/Research/datasets/extracted_plots/harv_2022/experiments/chm_add_azm_concat_no_queue',
    #      'ckpt': 'ckpts/exp_3/harv_10_channels_12_classes_swav_structure_patch_size_4_chm_add_azm_concat_no_queue_epoch=49.ckpt'
    #      },
    #       {'num_channels': 10,
    #      'num_classes': 12,
    #      'azm': False,
    #      'chm': True,
    #      'patch_size': 4,
    #      'log_every': 10,
    #      'max_epochs': 50,
    #      'num_workers': 4,
    #      'img_size': 40,
    #      'use_queue': False,
    #      'same_embed': False,
    #      'azm_concat': False,
    #      'chm_concat': True,
    #      'queue_chunks': 5,
    #      'extra_labels': 'chm_concat_only_no_queue',
    #       'save_dir': 'C:/Users/tonyt/Documents/Research/datasets/extracted_plots/harv_2022/experiments/chm_concat_only_no_queue',
    #      'ckpt': 'ckpts/exp_3/harv_10_channels_12_classes_swav_structure_patch_size_4_chm_concat_only_no_queue_epoch=49.ckpt'
    #      },
    #       {'num_channels': 10,
    #      'num_classes': 12,
    #      'azm': True,
    #      'chm': False,
    #      'patch_size': 4,
    #      'log_every': 10,
    #      'max_epochs': 50,
    #      'num_workers': 4,
    #      'img_size': 40,
    #      'use_queue': False,
    #      'same_embed': False,
    #      'azm_concat': True,
    #      'chm_concat': False,
    #      'queue_chunks': 5,
    #      'extra_labels': 'azm_concat_only_no_queue',
    #       'save_dir': 'C:/Users/tonyt/Documents/Research/datasets/extracted_plots/harv_2022/experiments/azm_concat_only_no_queue',
    #      'ckpt': 'ckpts/exp_3/harv_10_channels_12_classes_swav_structure_patch_size_4_azm_concat_only_no_queue_epoch=49.ckpt'
    #      },
    #       {'num_channels': 10,
    #      'num_classes': 60,
    #      'azm': True,
    #      'chm': True,
    #      'patch_size': 4,
    #      'log_every': 10,
    #      'max_epochs': 50,
    #      'num_workers': 4,
    #      'img_size': 40,
    #      'use_queue': False,
    #      'same_embed': False,
    #      'azm_concat': True,
    #      'chm_concat': True,
    #      'queue_chunks': 5,
    #      'extra_labels': 'all_concat_60_classes_no_queue',
    #       'save_dir': 'C:/Users/tonyt/Documents/Research/datasets/extracted_plots/harv_2022/experiments/all_concat_60_classes_no_queue',
    #      'ckpt': 'ckpts/exp_3/harv_10_channels_60_classes_swav_structure_patch_size_4_all_concat_60_classes_no_queue_epoch=49.ckpt'
    #      },
    #       {'num_channels': 10,
    #      'num_classes': 256,
    #      'azm': True,
    #      'chm': True,
    #      'patch_size': 4,
    #      'log_every': 10,
    #      'max_epochs': 50,
    #      'num_workers': 4,
    #      'img_size': 40,
    #      'use_queue': False,
    #      'same_embed': False,
    #      'azm_concat': True,
    #      'chm_concat': True,
    #      'queue_chunks': 5,
    #      'extra_labels': 'all_concat_256_classes_no_queue',
    #       'save_dir': 'C:/Users/tonyt/Documents/Research/datasets/extracted_plots/harv_2022/experiments/all_concat_256_classes_no_queue',
    #      'ckpt': 'ckpts/exp_3/harv_10_channels_256_classes_swav_structure_patch_size_4_all_concat_256_classes_no_queue_epoch=49.ckpt'
    #      },
    # ]

   
    
    # for config in configs:
    #     MODEL = models.SWaVModelStruct(**config).load_from_checkpoint(config['ckpt'],**config)
    #     try:
    #         valid.last_cluster = {}
    #         valid.do_plot_inference(config['save_dir'], MODEL)
    #     except KeyError as e:
    #         continue



    #valid.map_trees('C:/Users/tonyt/Documents/Research/datasets/extracted_plots/harv_2022/ind_plots_viz')

    #plot_inference(PLOT_PCA, 'test', MODEL)
    #valid.make_taxa_area_hists(PLOT_VIZ)
    #cluster_histograms('C:/Users/tonyt/Documents/Research/datasets/extracted_plots/harv_2022/pca_norm_clustered_plots', 'C:/Users/tonyt/Documents/Research/datasets/extracted_plots/harv_2022/pca_norm_clustered_plots/hists')
    #valid.make_taxa_plots('C:/Users/tonyt/Documents/Research/datasets/extracted_plots/harv_2022/taxon_plots')

    #cluster_plots(PLOT_PCA, 'C:/Users/tonyt/Documents/Research/datasets/extracted_plots/harv_2022/clustered_plots')
    #pca_norm_cluster_plots(PLOT_PCA, 'C:/Users/tonyt/Documents/Research/datasets/extracted_plots/harv_2022/pca_norm_clustered_plots')

    # valid.extract_plots(PLOT_PKLS)
    #handle_each_plot(PLOT_PKLS, get_shadow_masks, 'test')

    # inc_pca_plots(PLOT_PKLS, PLOT_PCA)


    # with open(PLOT_SUBSET, 'rb') as f:
    #     test = pickle.load(f)
    
    # print(test)
  
    # img = np.load(PCA)
    # img = torch.from_numpy(img).float()
    # img = rearrange(img, 'h w c -> c h w')
    # img = norm(img)
    # mp = torch.nn.MaxPool2d(3)
    # img = mp(img)
    # #img = img[3:7, ...]
    # test = torch.argmax(img, dim=0)
    # test = test.numpy()
    # #test = rearrange(test, 'c h w -> h w c')

    
    # # rgb = hp.pre_processing(IMG, wavelength_ranges=utils.get_viz_bands())
    # # rgb = hp.make_rgb(rgb["bands"])
    # # rgb = exposure.adjust_gamma(rgb, gamma=0.5)
    # # plt.imshow(rgb)
    # # plt.show()
    # for i in range(0, 30):
    #     #test[test != i] = 0
    #     #test[test == i] = 10
    #     plt.imshow(test == i)
    #     plt.show()
    # print(test)
   
    # print(len(valid.valid_data['taxonID'].unique()))
    # for key in valid.valid_files.keys():
    #     try:
    #         valid.validate(key, os.path.join(SAVE_DIR, f'{key}.npy'))
    #     except:
    #         print(key)

    

    # for spec in valid.valid_data['taxonID'].unique():
    #     plot_species(valid, spec)
    #     print('here')

    # for plot in valid.valid_data['plotID'].unique():
    #     side_by_side_bar(valid.confusion_matrix, plot)

    # print('here')

    # for k, f in valid.valid_files.items():
    #     try:
    #         img = np.load(f)
    #         img = torch.from_numpy(img).float()
    #         img = rearrange(img, 'h w c -> c h w')
    #         img = norm(img)
    #         test = torch.argmax(img, dim=0)
    #         test = test.numpy()
    #         np.save(os.path.join(SAVE_DIR, f'{k}.npy'), test)
    #     except:
    #         print(f)

    print('here')


    # for file in os.listdir(testing):
    #     show_file(os.path.join(testing,file))
    #     coords = file.split(".npy")[0]
    #     file = os.path.join(testing, file)
    #     valid.validate(coords, file)
    # #valid.confusion_matrix
    # for spec in valid.valid_data['taxonID'].unique():
    #     plot_species(valid, spec)
    #     print('here')
    # valid.validate('731000_4713000','validation/harv_simsiam_transformer_0_1/harv_transformer_60_classes_epoch=24/731000_4713000.npy')
    # print('here')
    # valid.confusion_matrix
    #valid.plot_trees(predictions = PRED_DIR)

    #bulk_validation(CKPTS_DIR, PCA_DIR, SAVE_DIR, VALID_FILE, site_name='HARV',num_channels=NUM_CHANNELS, num_classes=NUM_CLUSTERS, num_clusters=NUM_CLUSTERS, plot_file=PLOT_FILE, rearrange=False)

    # h5_file = "/data/shared/src/aalbanese/datasets/hs/NEON_refl-surf-dir-ortho-mosaic/NEON.D16.WREF.DP3.30006.001.2021-07.basic.20220330T192306Z.PROVISIONAL/NEON_D16_WREF_DP3_580000_5075000_reflectance.h5"
    # coords = "580000_5075000"
    # VALID = Validator(file="/data/shared/src/aalbanese/datasets/neon_wref.csv", img_dir=IMG_DIR, num_clusters=NUM_CLUSTERS)
    # checkpoint = "ckpts/saved/dense_sim_siam_30_channels_10_epochs/epoch=9.ckpt"

    # MODEL = inference.load_ckpt(models.DenseSimSiam, checkpoint, num_channels=NUM_CHANNELS, num_classes = NUM_CLUSTERS)
    # check_all(VALID, MODEL, SAVE_DIR, n_components=NUM_CHANNELS)
    # print(VALID.confusion_matrix)
    # VALID.confusion_matrix.to_csv(os.path.join(SAVE_DIR, "epoch_9_conf_matrix.csv"))


