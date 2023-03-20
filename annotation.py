import numpy as np
import geopandas as gpd
import pandas as pd
import h5py as hp
import shapely
import math
import os
from rasterio.transform import from_origin, AffineTransformer
from einops import rearrange
import rasterio as rs
import matplotlib.pyplot as plt
from rasterio.windows import Window
from skimage import morphology
from skimage.segmentation import watershed, mark_boundaries
from skimage.color import rgb2gray
from skimage.filters import sobel
from skimage.color import rgb2hsv
import argparse

class Plot:
    def __init__(
        self,
        utm_origin: tuple,
        width: int,
        rgb: np.ndarray,
        pca: np.ndarray,
        hyperspectral: np.ndarray,
        hyperspectral_bands: list,
        tree_tops: gpd.GeoDataFrame,
        canopy_height_model: np.ndarray,
        potential_trees: gpd.GeoDataFrame,
        epsg: str,
        base_dir: str,
        name: str,
        sitename: str,
        chm_dif_std: float
    ):
        self.name = name
        self.chm_dif_std = chm_dif_std
        self.sitename = sitename
        self.base_dir = base_dir
        self.epsg = epsg
        self.width = width
        self.utm_origin = utm_origin
        self.rgb = rgb
        self.pca = pca
        self.hyperspectral = hyperspectral
        self.hyperspectral_bands = hyperspectral_bands
        self.tree_tops = tree_tops.reset_index(drop=True)
        self.canopy_height_model = canopy_height_model
        self.potential_trees = potential_trees
        self.cm_affine = AffineTransformer(from_origin(self.utm_origin[0], self.utm_origin[1], .1, .1))
        self.m_affine = AffineTransformer(from_origin(self.utm_origin[0], self.utm_origin[1], 1, 1))
        #Rowcol calls yield y-x ordered coordinates tuple, we want x-y. [::-1] is the way to get a reverse view of a tuple, because python is elegant and pythonic
        self.tree_tops_local_cm = self.cm_affine.rowcol(self.tree_tops.geometry.x, self.tree_tops.geometry.y)[::-1]
        self.tree_tops_local_m = self.m_affine.rowcol(self.tree_tops.geometry.x, self.tree_tops.geometry.y)[::-1]

        self.filtered_trees = None

        self.identified_trees = list()


    def drop_ttops(self, include_idxs):
        self.tree_tops = self.tree_tops.iloc[include_idxs]

    def find_trees(self, algorithm):
        
        if algorithm == 'snapping':
            #This will modifiy some of the data in this object as well. They are intertwined like the forest and the sky
            #TODO: make them less intertwined
            tree_builder = TreeBuilderSnapping(self)
            if len(tree_builder.canopy_segments) >0:
                self.identified_trees = tree_builder.build_trees()
        
        if algorithm == 'scholl':
            tree_builder = TreeBuilderScholl(self)
            self.identified_trees = tree_builder.build_trees()

        if algorithm == 'filtering':
            tree_builder = TreeBuilderFiltering(self)
            self.identified_trees = tree_builder.build_trees()

    def manual_annotation(self):
        for tree in self.identified_trees:
            tp = TreePlotter(tree)

    def automatic_annotation(self):
        for tree in self.identified_trees:
            tree.save()

    def find_nearest(self, search_val):
        diff_arr = np.absolute(self.hyperspectral_bands-search_val)
        return diff_arr.argmin()

    def plot_me(self):
        tree_cm = self.cm_affine.rowcol(self.potential_trees.easting_tree, self.potential_trees.northing_tree)[::-1]
        tree_m = self.m_affine.rowcol(self.potential_trees.easting_tree, self.potential_trees.northing_tree)[::-1]


        fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(14, 4.25), layout="constrained")
        ax[0].imshow(self.rgb)
        ax[0].set_title('RGB - 10cm resolution')
        ax[0].scatter(*tree_cm, c='red')
        #ax[0].scatter(*self.tree_tops_local_cm, c='blue')
        ax[0].get_xaxis().set_visible(False)
        ax[0].get_yaxis().set_visible(False)

        ax[1].imshow(self.hyperspectral[...,[self.find_nearest(x) for x in [630,532,465]]]*12)
        ax[1].set_title('Hyperspectral - 1m resolution')
        ax[1].scatter(*tree_m, c='red')
        #ax[1].scatter(*self.tree_tops_local_m, c='blue')
        ax[1].get_xaxis().set_visible(False)
        ax[1].get_yaxis().set_visible(False)


        im = ax[2].imshow(self.canopy_height_model)
        ax[2].set_title('CHM - 1 m resolution')
        ax[2].get_xaxis().set_visible(False)
        ax[2].get_yaxis().set_visible(False)
        cbar = fig.colorbar(im)
        cbar.set_label('Height (m)')
        ax[2].scatter(*tree_m, c='red', label='Survey Tree')
        #ax[3].scatter(*self.tree_tops_local_m, c='blue', label='Treetop from Lidar')


        fig.suptitle(f"Plot ID: {self.name}\n")
        fig.legend(loc='upper right', bbox_to_anchor=(0.99,0.98))
        
        plt.show()

    def plot_before_and_after(self):
        tree_cm = self.cm_affine.rowcol(self.potential_trees.easting_tree, self.potential_trees.northing_tree)[::-1]
        filter_tree_cm = self.cm_affine.rowcol(self.filtered_trees.easting_tree, self.filtered_trees.northing_tree)[::-1]
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 4.25), layout="constrained")
        
        ax[0].imshow(self.rgb)
        ax[0].set_title('Tree Locations Before Filtering')
        ax[0].scatter(*tree_cm, c='red')
        #ax[0].scatter(*self.tree_tops_local_cm, c='blue')
        ax[0].get_xaxis().set_visible(False)
        ax[0].get_yaxis().set_visible(False)
        ax[0].scatter(*self.tree_tops_local_cm, c='blue')

        ax[1].imshow(self.rgb)
        ax[1].set_title('Tree Locations After Filtering')
        ax[1].scatter(*filter_tree_cm, c='red')
        #ax[0].scatter(*self.tree_tops_local_cm, c='blue')
        ax[1].get_xaxis().set_visible(False)
        ax[1].get_yaxis().set_visible(False)
        ax[1].scatter(*self.tree_tops_local_cm, c='blue')

        plt.show()


class Tree:
    def __init__(
        self,
        hyperspectral: np.ndarray,
        rgb: np.ndarray,
        rgb_mask: np.ndarray,
        hyperspectral_bands: np.ndarray,
        chm: np.ndarray,
        pca: np.ndarray,
        site_id: str,
        plot_id: str,
        utm_origin: tuple,
        individual_id:str,
        taxa: str,
        plot: Plot,
        algo_type: str,
        ):

        self.hyperspectral = hyperspectral
        self.rgb = rgb
        self.rgb_mask = rgb_mask
        self.hyperspectral_mask = self.make_hs_mask()
        self.hyperspectral_bands = hyperspectral_bands
        self.site_id = site_id
        self.taxa = taxa
        self.utm_origin = utm_origin
        self.plot_id = plot_id
        self.individual_id = individual_id
        self.chm = chm
        self.old_rgb_mask = None
        self.plot = plot
        self.algo_type = algo_type
        self.mpsi, self.ndvi = self.gather_filters()
        self.pca = pca

        self.name = f"{plot_id}_{individual_id}_{taxa}"

    def make_hs_mask(self):
        y_shape, x_shape = self.rgb_mask.shape[0]//10, self.rgb_mask.shape[1]//10
        out = np.zeros((y_shape, x_shape), dtype=np.bool8)
        for i in range(y_shape):
            for j in range(x_shape):
                subset = self.rgb_mask[i*10:(i+1)*10, j*10:(j+1)*10]
                if subset.sum()>50:
                    out[i, j] = True
        return out
    
    def clean_label_mask(self):
        self.old_rgb_mask = self.rgb_mask
        self.rgb_mask = morphology.remove_small_objects(morphology.erosion(self.rgb_mask), min_size=225)
        self.hyperspectral_mask = self.make_hs_mask()
    
    def gather_filters(self):

        rgb = self.hyperspectral[...,[self.plot.find_nearest(x) for x in [630,532,465]]]
        nir = self.hyperspectral[...,self.plot.find_nearest(750)]
        hsv = rgb2hsv(rgb)
        #Get Mixed Property Based Shadow Index (MPSI): (H- I) * (R - NIR)
        #Saturation is equivalent to intensity so using S from HSV
        mpsi = (hsv[:,:,0] - hsv[:,:,1]) * (rgb[:,:,0] - nir)
        ndvi = (nir - rgb[...,0])/(nir+rgb[...,0])
        return mpsi, ndvi


    def go_back_to_old_mask(self):
        self.rgb_mask = self.old_rgb_mask
        self.hyperspectral_mask = self.make_hs_mask()

    
    def save(self):
        savedir = os.path.join(self.plot.base_dir, self.algo_type, self.plot.name)
        if not os.path.exists(savedir):
            os.makedirs(savedir)
        np.savez(os.path.join(savedir, self.name),
            hyperspectral = self.hyperspectral,
            rgb = self.rgb,
            rgb_mask = self.rgb_mask,
            hyperspectral_mask = self.hyperspectral_mask,
            hyperspectral_bands = self.hyperspectral_bands,
            chm = self.chm,
            utm_origin = np.array(self.utm_origin),
            taxa = self.taxa,
            plot_id = self.plot_id,
            site_id = self.site_id,
            algo_type = self.algo_type,
            mpsi = self.mpsi,
            ndvi = self.ndvi,
            pca = self.pca
            )


class TileSet:
    ##ASSUMES EVERYTHING IS A LOWER LEFT ORIGIN, BECAUSE THAT IS THE NEON CONVENTION
    def __init__(
        self,
        tile_dir: str,
        epsg: str,
        file_ext: str,
        coord_locs: tuple,
        file_width: int = 1000, 
    ):
        self.all_files = [f for f in os.scandir(tile_dir) if f.is_file() and f.path.endswith(file_ext)]
        self.epsg = epsg
        self.coord_locs = coord_locs
        #Assumes square file size.
        self.file_width = file_width

        self.tile_gdf = self.__make_tile_gdf__()


    def __make_tile_gdf__(self):
        polygons = []
        file_west_bounds = []
        file_north_bounds = []
        #Using filenames instead of actual metadata since metadata we are dealing with a bunch of different filetypes
        #Filenames: The original metadata
        for f in self.all_files:
            split_name = f.path.split('_')
            min_x, min_y = int(split_name[self.coord_locs[0]]), int(split_name[self.coord_locs[1]])
            max_x, max_y = min_x + self.file_width, min_y + self.file_width
            file_west_bounds.append(min_x)
            file_north_bounds.append(max_y)
            tile_poly = shapely.box(min_x, min_y, max_x, max_y)
            polygons.append(tile_poly)

        gdf = gpd.GeoDataFrame(
            data={
                    'filepath': self.all_files,
                    'file_west_bound': file_west_bounds,
                    'file_north_bound': file_north_bounds
                }, 
            geometry=polygons, 
            crs=self.epsg)

        return gdf


class PlotBuilder:
    """Takes in known data for a study site and returns Plot object populated with relevant data for that plot. 
        All plots should be contained within a Study Site object"""
    def __init__(
        self,
        sitename: str,
        epsg: str,
        base_dir: str,
        completed_plots: list = [],
        plot_hs_dif = False
        ):
        #Static Vars
        self.base_dir = os.path.join(base_dir, sitename)
        self.sitename = sitename
        self.epsg = epsg
        self.h5_tiles = TileSet(os.path.join(self.base_dir, "HS"), epsg, '.h5', (-3,-2))
        self.pca_tiles = TileSet(os.path.join(self.base_dir, "PCA"), epsg, '.npy', (-4,-3))
        self.chm_tiles = TileSet(os.path.join(self.base_dir, "CHM"), epsg, '.tif', (-3, -2))
        self.ttop_file = gpd.read_file(os.path.join(self.base_dir, f'{sitename}_tree_tops.gpkg'))
        self.rgb_tiles = TileSet(os.path.join(self.base_dir, "RGB"), epsg, '.tif', (-3, -2))
        temp_df = pd.read_csv(os.path.join(self.base_dir,f'{sitename}_woody_vegetation.csv'))
        self.plot_data_file = gpd.GeoDataFrame(
                                temp_df,
                                crs= epsg,
                                geometry=gpd.points_from_xy(temp_df['easting_tree'], temp_df['northing_tree'])
                                ).sort_values(['easting_plot', 'northing_plot'])
        print(f'loaded species information with the following species counts\n{temp_df["taxonID"].value_counts()}')
        
        self.all_plot_ids = sorted(list(set(np.unique(self.plot_data_file.plotID)) - set(completed_plots)))
        self.hs_filters = [[410,1320],[1450,1800],[2050,2485]]
        self.plot_hs_dif = plot_hs_dif
        self.chm_dif_std = self.plot_data_file['chm_dif'].std()

    def get_hs_filter(self, bands):
        # hs_filter should be a list of [min, max]
        mask_list = [(bands>=lmin) & (bands<=lmax) for lmin, lmax in self.hs_filters]
        band_mask = np.logical_or.reduce(mask_list)
        idxs = np.where(band_mask)[0]
        return idxs

    def build_plots(self):
        for plot_id in self.all_plot_ids:
            yield self.__build_plot__(plot_id)
    
    #If only we could have private methods...
    def __build_plot__(self, plot_id):
        #Select the relevant data
        selected_plot = self.plot_data_file.loc[self.plot_data_file.plotID == plot_id]


        first_row = selected_plot.iloc[0]
        #Need to fit things to the nearest 1m since the hyperspectral data is 1m
        #Centroid is floored instead of rounded for consistency with affine transformer
        plot_centroid = math.floor(first_row.easting_plot), math.floor(first_row.northing_plot)
        #plot_width = first_row.plotSize ** (1/2)
        #Max plot width is 40 so we'll just use that instead of trying to deal with variable plot sizes
        plot_width = 40
        min_x, min_y = plot_centroid[0] - (plot_width//2), plot_centroid[1] - (plot_width//2)
        max_x, max_y = min_x + plot_width, min_y + plot_width

        plot_bbox = shapely.box(min_x, min_y, max_x, max_y)

        hs, hs_bands = self.grab_hs(plot_bbox)
        assert hs.shape[0] == plot_width and hs.shape[1] == plot_width, 'hyperspectral plot does not match plot dims'
        
        chm = self.grab_chm(plot_bbox)
        assert chm.shape[0] == plot_width and chm.shape[1] == plot_width, 'chm plot does not match plot dims'

        pca = self.grab_pca(plot_bbox)
        assert chm.shape[0] == plot_width and chm.shape[1] == plot_width, 'chm plot does not match plot dims'
        
        rgb = self.grab_rgb(plot_bbox)
        assert rgb.shape[0] == plot_width*10 and rgb.shape[1] == plot_width*10, 'rgb plot does not match plot dims'
        
        ttops = self.grab_ttops(plot_bbox)

        origin = min_x, max_y

        return Plot(
            utm_origin=origin,
            width = plot_width,
            rgb = rgb,
            pca=pca,
            hyperspectral= hs,
            hyperspectral_bands= hs_bands,
            tree_tops= ttops,
            canopy_height_model= chm,
            potential_trees= selected_plot,
            epsg = self.epsg,
            base_dir= self.base_dir,
            name = plot_id,
            sitename=self.sitename,
            chm_dif_std=self.chm_dif_std
        )



    def get_crop_values(self, west_bound, north_bound, scale, tile_bounds):
        affine = AffineTransformer(from_origin(west_bound, north_bound, scale, scale))

        max_y, min_x = affine.rowcol(tile_bounds[0], tile_bounds[1])
        min_y, max_x = affine.rowcol(tile_bounds[2], tile_bounds[3])

        return (min_x, min_y, max_x, max_y)

    def grab_hs(self, bbox):
        hs_grabs = []
        bounds_list = []
        tiles = self.__get_relevant_entries__(bbox, self.h5_tiles)
        for ix, tile in tiles.iterrows():
            min_x, min_y, max_x, max_y = self.get_crop_values(tile.file_west_bound, tile.file_north_bound, 1, tile.geometry.bounds)
            bounds_list.append((tile.file_west_bound, tile.file_north_bound))
            hs_file = hp.File(tile.filepath.path, 'r')

            bands = hs_file[self.sitename]["Reflectance"]["Metadata"]['Spectral_Data']['Wavelength'][:]

            hs_filter = self.get_hs_filter(bands)
            hs_grab = hs_file[self.sitename]["Reflectance"]["Reflectance_Data"][min_y:max_y,min_x:max_x,...]/10000
            if self.plot_hs_dif:
                self.plot_hs_spectra(bands, hs_filter, hs_grab)
            hs_grab = hs_grab[...,hs_filter]
            bands = bands[hs_filter]
            
            
            hs_grab = hs_grab.astype(np.float32)
            #TODO: Clean up values over/under 1
            hs_grabs.append(hs_grab)
            hs_file.close()

        if len(hs_grabs) == 1:
            return hs_grabs[0], bands
        else:
            return self.__concat_plots__(hs_grabs, bounds_list), bands

    def plot_hs_spectra(self, bands, hs_filter, hs_grab):
        mean_1 = np.mean(hs_grab, axis=(0,1))
        mean_2 = np.mean(hs_grab[...,hs_filter], axis=(0,1))
        bands_2 = bands[hs_filter]

        fig, ax = plt.subplots(2, 1, figsize=(8, 6))
        ax[0].set_ylabel('Reflectance', loc="bottom")
        ax[1].set_xlabel('Wavelength (nm)')
        ax[0].plot(bands, mean_1, '.--b')
        ax[1].plot(bands_2, mean_2, '.--g')
        ax[0].set_title("Before de-noising")

        ax[0].set_ylim(-0.01, 0.4)
        ax[1].set_ylim(-0.01, 0.4)
        ax[1].set_title("After de-noising")
        plt.suptitle("Mean hyperspectral reflectance from a sample plot before and after de-noising")
        fig.tight_layout()
        plt.show()

    def grab_chm(self, bbox) -> np.ndarray:
        chm_grabs = []
        bounds_list = []
        tiles = self.__get_relevant_entries__(bbox, self.chm_tiles)
        for ix, tile in tiles.iterrows():
            min_x, min_y, max_x, max_y = self.get_crop_values(tile.file_west_bound, tile.file_north_bound, 1, tile.geometry.bounds)
            bounds_list.append((tile.file_west_bound, tile.file_north_bound))
            chm = rs.open(tile.filepath.path)
            chm_grab = chm.read(1, window=Window.from_slices((min_y, max_y), (min_x, max_x)))
            
            chm_grab[chm_grab<0] = 0
            chm_grabs.append(chm_grab)
            chm.close()
        
        if len(chm_grabs) == 1:
            return chm_grabs[0]
        else:
            return self.__concat_plots__(chm_grabs, bounds_list)
        
    def grab_pca(self, bbox) -> np.ndarray:
        pca_grabs = []
        bounds_list = []
        tiles = self.__get_relevant_entries__(bbox, self.pca_tiles)
        for ix, tile in tiles.iterrows():
            min_x, min_y, max_x, max_y = self.get_crop_values(tile.file_west_bound, tile.file_north_bound, 1, tile.geometry.bounds)
            bounds_list.append((tile.file_west_bound, tile.file_north_bound))
            pca = np.load(tile.filepath.path, mmap_mode='r')[min_y:max_y,min_x:max_x,...]
            
            pca_grabs.append(pca)
        
        if len(pca_grabs) == 1:
            return pca_grabs[0]
        else:
            return self.__concat_plots__(pca_grabs, bounds_list)
    
    def grab_rgb(self, bbox) -> np.ndarray:
        rgb_grabs = []
        bounds_list = []
        tiles = self.__get_relevant_entries__(bbox, self.rgb_tiles)
        for ix, tile in tiles.iterrows():
            min_x, min_y, max_x, max_y = self.get_crop_values(tile.file_west_bound, tile.file_north_bound, .1, tile.geometry.bounds)
            bounds_list.append((tile.file_west_bound, tile.file_north_bound))
            rgb = rs.open(tile.filepath.path)
            rgb_grab = rgb.read(window=Window.from_slices((min_y, max_y), (min_x, max_x)))
            
            rgb_grab = rearrange(rgb_grab, 'c h w -> h w c')
            rgb_grabs.append(rgb_grab)
            rgb.close()
        
        if len(rgb_grabs) == 1:
            return rgb_grabs[0]
        else:
            return self.__concat_plots__(rgb_grabs, bounds_list)
    
    def grab_ttops(self, bbox):
        return self.ttop_file.clip(bbox)

    def __concat_plots__(self, plots_list, bounds_list):
        if len(bounds_list) == 2:
            west_dif = (bounds_list[1][0] - bounds_list[0][0])//1000
            north_dif = (bounds_list[1][1] - bounds_list[0][1])//1000

            concat_axis = 1 if west_dif != 0 else 0

            return np.concatenate(plots_list, axis=concat_axis)
        elif len(bounds_list) == 4:

            pass


    def __get_relevant_entries__(self, bbox: shapely.Polygon, tileset: TileSet) -> gpd.GeoDataFrame:
        return tileset.tile_gdf.clip(bbox).sort_values(['file_west_bound', 'file_north_bound'])


class CanopySegment:
    def __init__(
        self,
        bbox: np.ndarray,
        affine,
        epsg,
        original_index
    ):

        self.x_min, self.y_min, self.x_max, self.y_max = self.get_bounds(bbox)
        self.epsg = epsg
        self.ttop_index = original_index - 1
        #Origin = upper left, row-col
        #Anything local is y-x
        #Anything utm is x-y
        #If only there was some way to standardize this!
        self.local_origin = self.y_min, self.x_min

        self.utm_origin = affine.xy(*self.local_origin)
        self.utm_x_min, self.utm_y_min = affine.xy(self.y_min, self.x_min)
        self.utm_x_max, self.utm_y_max = affine.xy(self.y_max, self.x_max)

    def get_bounds(self, bbox: np.ndarray):
        bbox_min = bbox.min(0)
        bbox_max = bbox.max(0)

        return bbox_min[1], bbox_min[0], bbox_max[1], bbox_max[0]
    
    def to_polygon(self):
        return shapely.box(self.utm_x_min, self.utm_y_min, self.utm_x_max, self.utm_y_max)

    def to_dict(self):
        return {'properties':{
                    'ttop_index': self.ttop_index,
                    'epsg': self.epsg,
                    'utm_origin': self.utm_origin,
                    'local_x_min': self.x_min,
                    'local_y_min': self.y_min,
                    'local_x_max': self.x_max,
                    'local_y_max': self.y_max
                }, 
                'geometry': self.to_polygon()}

class TreeBuilderScholl:
    def __init__(self, plot: Plot):
        self.plot = plot


class TreeBuilderFiltering:
    def __init__(self, plot: Plot):
        self.plot = plot
        self.filtered_trees = self.filter_trees()

    def filter_trees(self) -> gpd.GeoDataFrame:
        return self.plot.potential_trees.loc[self.plot.potential_trees['chm_dif']<self.plot.chm_dif_std].reset_index(drop=True)
        #return self.plot.potential_trees.reset_index(drop=True)
    
    def remove_close_trees(self):
        dist_matrix = self.filtered_trees.geometry.apply(lambda g: self.filtered_trees.distance(g)).to_numpy()
        upper_tri = dist_matrix[np.triu_indices_from(dist_matrix, k=1)]
        thresh = upper_tri.mean() - upper_tri.std()*1.5
        #Get index pairs where trees are suspiciously close
        sus_indexes = set(tuple(sorted(x)) for x in np.argwhere(dist_matrix<thresh))
        to_drop = list()
        for sus in sus_indexes:
            if sus[0] == sus[1]:
                continue

            height_0 = self.filtered_trees.loc[sus[0]]['chm_height']
            height_1 = self.filtered_trees.loc[sus[1]]['chm_height']

            to_add = sus[0] if height_1>=height_0 else sus[1]
            to_drop.append(to_add)

        to_drop = set(to_drop)
        self.filtered_trees = self.filtered_trees.drop(to_drop).reset_index(drop=True)
        self.plot.filtered_trees = self.filtered_trees


    def build_trees(self):
        self.remove_close_trees()
        trees = list()
        for ix, tree in self.filtered_trees.iterrows():
            taxa = tree.taxonID
            plot_id = tree.plotID
            individual_id = tree.individualID
            site_id = self.plot.sitename

            #y-x
            cm_loc = self.plot.cm_affine.rowcol(tree.easting_tree, tree.northing_tree)
            m_loc = self.plot.m_affine.rowcol(tree.easting_tree, tree.northing_tree)

            m_buffer = 2
            cm_buffer = 20

            y_m_min, y_m_max, x_m_min, x_m_max = m_loc[0] - m_buffer, m_loc[0]+m_buffer, m_loc[1]-m_buffer, m_loc[1]+m_buffer
            y_cm_min, y_cm_max, x_cm_min, x_cm_max = cm_loc[0] - cm_buffer, cm_loc[0]+cm_buffer, cm_loc[1]-cm_buffer, cm_loc[1]+cm_buffer
            utm_origin = tree.easting_tree - m_buffer, tree.northing_tree + m_buffer

            rgb = self.plot.rgb[y_cm_min:y_cm_max, x_cm_min:x_cm_max,...]
            chm = self.plot.canopy_height_model[y_m_min:y_m_max, x_m_min:x_m_max]
            hs = self.plot.hyperspectral[y_m_min:y_m_max, x_m_min:x_m_max,...]
            pca = self.plot.pca[y_m_min:y_m_max, x_m_min:x_m_max,...]
            hs_bands = self.plot.hyperspectral_bands

            rgb_mask = np.ones(shape=(rgb.shape[0],rgb.shape[1]), dtype=np.bool8)

            new_tree = Tree(
                hyperspectral=hs,
                hyperspectral_bands=hs_bands,
                rgb=rgb,
                rgb_mask=rgb_mask,
                chm=chm,
                site_id=site_id,
                plot_id=plot_id,
                utm_origin=utm_origin,
                individual_id=individual_id,
                taxa=taxa,
                plot=self.plot,
                algo_type="distance_filtering",
                pca=pca
            )
            trees.append(new_tree)

        return trees

class TreeBuilderSnapping:
    def __init__(self, plot: Plot):
        self.plot = plot
        #Find visible canopies using watershed segmentation + lidar treetops locations
        self.canopy_segments, self.labelled_plot = self.segment_canopy()
        if len(self.canopy_segments) > 0: 
            #Make geodataframe of segment bounding boxes
            #Need to get actual label geometry in here somehow
            self.canopy_segments_gdf = gpd.GeoDataFrame.from_features([cs.to_dict() for cs in self.canopy_segments])

            #Drop any tree tops that didn't match visible trees
            self.plot.drop_ttops(self.canopy_segments_gdf['ttop_index'])
            self.tree_crown_pairs = self.identify_trees()
            #self.labelled_trees = self.build_trees()
    
    def build_trees(self):
        trees = []
        #For each tree crown pair, assemble all data into Tree object
        for tree_idx, crown_idx in self.tree_crown_pairs.items():
            selected_tree = self.plot.potential_trees.loc[tree_idx]
            selected_crown = self.canopy_segments_gdf.loc[self.canopy_segments_gdf['ttop_index'] == crown_idx]

            taxa = selected_tree.taxonID
            plot_id = selected_tree.plotID
            individual_id = selected_tree.individualID
            site_id = selected_tree.siteID

            local_y_min, local_y_max = int(selected_crown.local_y_min), int(selected_crown.local_y_max)
            local_x_min, local_x_max = int(selected_crown.local_x_min), int(selected_crown.local_x_max)

            rgb = self.plot.rgb[local_y_min:local_y_max, local_x_min:local_x_max,...]
            chm = self.plot.canopy_height_model[local_y_min:local_y_max, local_x_min:local_x_max,...]

            label_subset = self.labelled_plot[local_y_min:local_y_max, local_x_min:local_x_max,...]
            label_mask = label_subset == crown_idx +1

            hs_x_min, hs_y_min = math.floor(local_x_min/10), math.floor(local_y_min/10)
            hs_x_max, hs_y_max = math.ceil(local_x_max/10), math.ceil(local_y_max/10)

            x_left_pad, x_right_pad = local_x_min-(hs_x_min*10), (hs_x_max*10) - local_x_max
            y_up_pad, y_down_pad = local_y_min - (hs_y_min*10), (hs_y_max*10) - local_y_max 

            rgb_mask = np.pad(label_mask, ((y_up_pad, y_down_pad), (x_left_pad, x_right_pad)))
            chm = np.pad(chm, ((y_up_pad, y_down_pad), (x_left_pad, x_right_pad)))
            rgb = np.pad(rgb, ((y_up_pad, y_down_pad), (x_left_pad, x_right_pad), (0,0)))

            hs = self.plot.hyperspectral[hs_y_min:hs_y_max, hs_x_min:hs_x_max, ...]
            hs_bands = self.plot.hyperspectral_bands
            
            #Fix origin to account for padding
            #0 is y, 1 is x
            utm_origin = selected_crown.utm_origin.iat[0]
            utm_origin = utm_origin[0] + y_up_pad*.1, utm_origin[1] - x_left_pad*.1
            

            new_tree = Tree(
                hyperspectral=hs,
                rgb=rgb,
                rgb_mask=rgb_mask,
                hyperspectral_bands=hs_bands,
                chm=chm,
                site_id=site_id,
                plot_id=plot_id,
                utm_origin=utm_origin,
                individual_id=individual_id,
                taxa=taxa,
                plot=self.plot,
                algo_type="snapping"
            )
            trees.append(new_tree)


        return trees
    


    def segment_canopy(self):
        mask = self.plot.canopy_height_model>2

        #CAN ALSO TRY SCIPY METHODS HERE
        mask = morphology.remove_small_holes(mask)
        mask = morphology.remove_small_objects(mask)

        markers = np.zeros(self.plot.rgb.shape[0:2], dtype=np.int32)
        for ix, rowcol in enumerate(self.plot.tree_tops_local):
            #Zero means not a marker so we add 1
            markers[rowcol] = ix + 1

        labelled = watershed(sobel(rgb2gray(self.plot.rgb)), markers=markers, mask=mask, compactness=0.01)
        #TODO: downsample this to work with 1m chm

        canopy_segs = []
        for ix in np.unique(labelled):
            if ix == 0:
                continue
            bbox = np.argwhere(labelled == ix)
            canopy_segs.append(CanopySegment(bbox, self.plot.cm_affine, self.plot.epsg, ix))

        return canopy_segs, labelled
    
    def identify_trees(self, search_buffer = 3, max_search = 10):
        tree_skip_list = set()
        selected_crowns = set()
        labelled_pairs = dict()
        searches = 0
        num_pairs = len(labelled_pairs)
        while searches<max_search:
            for ix, tree in self.plot.potential_trees.iterrows():
                if ix not in tree_skip_list:
                    #Finds the index of the best tree top/crown pair based on distance to tree top
                    best_crown_idx = self.find_best_crown(tree, search_buffer, selected_crowns)
                    #Finds the best potential tree for that tree top/crown pair
                    if best_crown_idx is not None:
                        best_tree_idx = self.find_best_tree(best_crown_idx, search_buffer, tree_skip_list)
                        if best_tree_idx == ix:
                            selected_crowns.add(best_crown_idx)
                            tree_skip_list.add(best_tree_idx)
                            labelled_pairs[int(best_tree_idx)] = int(best_crown_idx)
                    else:
                        #If there are no treetops within distance just throw this one out
                        tree_skip_list.add(ix)
                    #If they are both each others best pair, add them to the pairs list and remove from consideration
                    
            searches += 1
            #Early stopping if we are not adding anymore pairs
            if num_pairs == len(labelled_pairs):
                break
            num_pairs = len(labelled_pairs)
        
        return labelled_pairs
        
        print('here')
                

    #Find the closest tree top/crown pair based on distance to potential labelled tree
    def find_best_crown(self, tree, search_buffer, selected_crowns):
        #Remove any already selected tops/crowns from consideration
        test_tops = self.plot.tree_tops.loc[self.plot.tree_tops.index.difference(selected_crowns)]
        distances = test_tops.distance(tree.geometry)
        distances = distances[distances<search_buffer]
        if len(distances) == 0:
            return None
        return distances.sort_values().index[0]

    #Find best potential labelled tree based on distance to tree top/crown pair
    def find_best_tree(self, best_crown_idx, search_buffer, tree_skip_list):

        tree_top = self.plot.tree_tops.loc[best_crown_idx]
        #Remove any skipped/selected trees from consideration
        test_trees = self.plot.potential_trees.loc[self.plot.potential_trees.index.difference(tree_skip_list)]
        distances = test_trees.distance(tree_top.geometry)
        distances = distances[distances<search_buffer]
        if len(distances) == 0:
            return None
        return distances.sort_values().index[0]


class TreePlotter:
    def __init__(
        self,
        tree: Tree,
        #save_size: int
    ):
        self.tree = tree
        #self.save_size = save_size

        self.fig, self.axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
        self.hs_ax = self.axes[0]
        self.rgb_ax = self.axes[1]
        self.big_rgb_ax = self.axes[2]

        self.hs_ax.set_title('1m Hyperspectral Mask')
        self.rgb_ax.set_title('10cm RGB')
        self.big_rgb_ax.set_title('Full Plot')
        self.draw_big_rgb()

        self.rgb_ticks_x = np.arange(0, self.tree.rgb.shape[1], 10)
        self.rgb_ticks_y = np.arange(0, self.tree.rgb.shape[0], 10)

        self.hs_im = self.draw_hs()
        self.rgb_im = self.draw_rgb()

        self.fig.canvas.mpl_connect('key_press_event', self.on_press)
        self.fig.canvas.mpl_connect('pick_event', self.on_click)
        self.fig.suptitle("A = Accept and Save | R = Reject | C = Clean Up | V = Revert\nClick to toggle pixels")
        self.fig.supxlabel(tree.name)

        plt.show()
        print('here')
        
    def on_press(self, event):
        #print('press')
        if event.key == 'c':
            self.tree.clean_label_mask()
            self.update()
        if event.key == 'v':
            self.tree.go_back_to_old_mask()
            self.update()
        if event.key == 'a':
            self.tree.save()
            plt.close()
        if event.key == 'r':
            plt.close()
        if event.key == 's':
            self.fig.savefig(os.path.join(self.tree.plot.base_dir, "Figures", "Annotation", self.tree.name + ".pdf"))

    def on_click(self, event):
        #print('event')
        artist = event.artist
        if artist.axes == self.hs_ax:
            self.handle_hs_click(event)
        if artist.axes == self.rgb_ax:
            self.handle_rgb_click(event)
        self.update()

    def update(self):
        self.hs_ax.clear()
        self.rgb_ax.clear()
        self.draw_hs()
        self.draw_rgb()
        self.hs_im.axes.figure.canvas.draw()
        self.rgb_im.axes.figure.canvas.draw()

    def draw_hs(self):
        
        hs_im = self.hs_ax.imshow(self.tree.hyperspectral_mask, picker=True)
        return hs_im

    def draw_rgb(self):
        
        rgb_im = self.rgb_ax.imshow(mark_boundaries(self.tree.rgb, self.tree.rgb_mask), picker=True)
        self.rgb_ax.set_xticks(self.rgb_ticks_x)
        self.rgb_ax.set_yticks(self.rgb_ticks_y)
        self.rgb_ax.grid()

        return rgb_im
    
    def draw_big_rgb(self):
        self.big_rgb_ax.imshow(self.tree.plot.rgb)
        orig_tree = self.tree.plot.filtered_trees.loc[self.tree.plot.filtered_trees['individualID'] == self.tree.individual_id]
        tree_loc = self.tree.plot.cm_affine.rowcol(orig_tree.easting_tree, orig_tree.northing_tree)[::-1]
        self.big_rgb_ax.scatter(*tree_loc)
    
    def find_nearest(self, search_val):
        diff_arr = np.absolute(self.tree.hyperspectral_bands-search_val)
        return diff_arr.argmin()

    def handle_hs_click(self, event):
        x_loc = round(event.mouseevent.xdata)
        y_loc = round(event.mouseevent.ydata)
        #print(y_loc)
        self.tree.hyperspectral_mask[y_loc, x_loc] = ~self.tree.hyperspectral_mask[y_loc, x_loc]
    
    def handle_rgb_click(self, event):
        x_loc = math.floor(event.mouseevent.xdata/10)
        y_loc = math.floor(event.mouseevent.ydata/10)
        self.tree.hyperspectral_mask[y_loc, x_loc] = ~self.tree.hyperspectral_mask[y_loc, x_loc]

    
def annotate_all(**kwargs):
    pb = PlotBuilder(**kwargs)
    for plot in pb.build_plots():
        pass
        #plot.chm_filter()
        #plot.find_trees()
        #plot.plot_and_check_trees()

    


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("-s", "--sitename", help='NEON sitename, e.g. NIWO')
    # parser.add_argument("-b", "--basedir", help="Base directory storing all NEON data")
    # parser.add_argument("-e", "--epsg", help='EPSG code, e.g EPSG:32613')
    # parser.add_argument("-algo", help="Tree selection algorithm to use. One of: filtering, snapping, scholl")
    # parser.add_argument("-m", "--manual", help="perform manual annotation",
    #                 action="store_true")
    # parser.add_argument("-a", "--automatic", help="perform automatic annotation",
    #                 action="store_true")
    # parser.add_argument("--skip-plots", help="Any plots from a study site you may want to skip, separated by spaces, eg. NIWO_057 NIWO_019")
    # parser.add_argument()
    # args = parser.parse_args()
    #TODO: Implement parser to run these commands
        #Sitename
        #Basedir
        #Annotation Style
        #Completed plots

    test = PlotBuilder(
        sitename='NIWO',
        epsg='EPSG:32613',
        base_dir=r'C:\Users\tonyt\Documents\Research\final_data'
    )

    niwo_57 = test.__build_plot__('NIWO_057')
    niwo_57.find_trees('filtering')
    niwo_57.plot_before_and_after()
    niwo_57.automatic_annotation()
    print('here')
    

    # annotate_all(
    #     completed_plots = ['NIWO_001', 'NIWO_002', 'NIWO_004', 'NIWO_005', 'NIWO_007', 'NIWO_009', 'NIWO_010', 'NIWO_011', 'NIWO_012', 'NIWO_014', 'NIWO_015', 'NIWO_016', 'NIWO_017', 'NIWO_041'],
    #     sitename = "NIWO",
    #     h5_files= 'W:/Classes/Research/datasets/hs/original/NEON.D13.NIWO.DP3.30006.001.2020-08.basic.20220516T164957Z.RELEASE-2022',
    #     chm_files= 'C:/Users/tonyt/Documents/Research/datasets/chm/niwo_valid_sites_test',
    #     ttop_files = "C:/Users/tonyt/Documents/Research/datasets/niwo_tree_tops",
    #     tree_data_file= 'C:/Users/tonyt/Documents/Research/datasets/tree_locations/NIWO_by_me.geojson',
    #     rgb_files =  r'C:\Users\tonyt\Documents\Research\datasets\rgb\NEON.D13.NIWO.DP3.30010.001.2020-08.basic.20220814T183511Z.RELEASE-2022',
    #     epsg='EPSG:32613',
    #     base_dir=r'C:\Users\tonyt\Documents\Research\thesis_final'
    #     )