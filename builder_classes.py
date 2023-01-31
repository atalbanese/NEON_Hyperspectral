import numpy as np
import geopandas as gpd
import pandas as pd
import h5py as hp
import shapely
import torch
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


class Tree:
    def __init__(
        self,
        hyperspectral: np.ndarray,
        rgb: np.ndarray,
        rgb_mask: np.ndarray,
        hyperspectral_mask: np.ndarray,
        hyperspectral_bands: list,
        site_id: str,
        tile_origin: tuple,
        utm_origin: tuple,
        taxa: str = None,
        ):

        self.hyperspectral = hyperspectral
        self.rgb = rgb
        self.rgb_mask = rgb_mask
        self.hyperspectral_mask = hyperspectral_mask
        self.hyperspectral_bands = hyperspectral_bands
        self.site_id = site_id
        self.taxa = taxa
        self.utm_origin = utm_origin
        self.tile_origin = tile_origin

    def get_masked_hs(self, out_type ='numpy'):
        if out_type == 'numpy':
            return self.hyperspectral[self.hyperspectral_mask]
        if out_type == 'torch':
            return torch.from_numpy(self.hyperspectral[self.hyperspectral_mask])


    def get_masked_rgb(self, out_type ='numpy'):
        if out_type == 'numpy':
            return self.rgb[self.rgb]
        if out_type == 'torch':
            return torch.from_numpy(self.rgb[self.rgb_mask])


class Plot:
    def __init__(
        self,
        utm_origin: tuple,
        width: int,
        rgb: np.ndarray,
        hyperspectral: np.ndarray,
        hyperspectral_bands: list,
        tree_tops: gpd.GeoDataFrame,
        canopy_height_model: np.ndarray,
        potential_trees: list
    ):
        self.width = width
        self.utm_origin = utm_origin
        self.rgb = rgb
        self.hyperspectral = hyperspectral
        self.hyperspectral_bands = hyperspectral_bands
        self.tree_tops = tree_tops
        self.canopy_height_model = canopy_height_model
        self.potential_trees = potential_trees
        self.tree_tops_local = self.make_local_tree_tops()

        self.identified_trees = list()
    
    def make_local_tree_tops(self):
        affine = AffineTransformer(from_origin(self.utm_origin[0], self.utm_origin[1], .1, .1))
        out_list = []
        for ix, tree in self.tree_tops.iterrows():
            out_list.append(affine.rowcol(tree.geometry.x, tree.geometry.y))
        return out_list

    def find_trees(self):
        tree_builder = TreeBuilder(self)
        identified_trees = tree_builder.identify_trees()

        


    pass


class TileSet:
    ##ASSUMES EVERYTHING IS A LOWER LEFT ORIGIN, BECAUSE THAT IS THE NEON CONVENTION
    ##I AM PERSONALLY MORE OF AN UPPER LEFT ORIGIN GUY SINCE THAT IS THE WAY NUMPY ARRAYS ARE ORDERED 
    def __init__(
        self,
        tile_dir: str,
        epsg: str,
        file_ext: str,
        coord_locs: tuple,
        file_width: int, 
    ):
        self.all_files = [f for f in os.scandir(tile_dir) if f.is_file() and f.path.endswith(file_ext)]
        self.epsg = epsg
        self.coord_locs = coord_locs
        #Assumes square file size. Do we need to know units? 1m vs 10cm
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
            max_x, max_y = min_x + 1000, min_y + 1000
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
        All plots should be contained within a Study Site object. The goal is total abstraction"""
    def __init__(
        self,
        sitename: str,
        h5_files: str,
        chm_files: str,
        ttop_files: str,
        rgb_files: str,
        tree_data_file: str,
        epsg: str
        ):
        #Static Vars
        self.sitename = sitename
        self.h5_tiles = TileSet(h5_files, epsg, '.h5', (-3,-2), 1000)
        self.chm_tiles = TileSet(chm_files, epsg, '.tif', (-3, -2), 1000)
        self.ttop_tiles = TileSet(ttop_files, epsg, '.geojson', (-3, -2), 1000)
        self.rgb_tiles = TileSet(rgb_files, epsg, '.tif', (-3, -2), 1000)
        self.plot_data_file = gpd.read_file(tree_data_file).sort_values(['easting_plot', 'northing_plot'])
        self.all_plot_ids = list(np.unique(self.plot_data_file.plotID))

        #Will need two affine transforms for 1m and 10cm raster data

        #Mutable Vars
        self.current_plot_id = None
        #Caching tree tops files since the other file types can be partially opened
        self.open_tops = None
        self.open_tops_file = ''

    def build_plots(self):
        for plot_id in self.all_plot_ids:
            self.current_plot_id = plot_id
            yield self.__build_plot__()
    
    #If only we could have private methods...
    def __build_plot__(self):
        #Select the relevant data
        selected_plot = self.plot_data_file.loc[self.plot_data_file.plotID == self.current_plot_id]

        #Need to get: Hyperspectral, Canopy Height Model, RGB, Treetops
        #There are plots that cross file boundaries, this will be the tricky part. We should use gis operations to do as much as possible

        first_row = selected_plot.iloc[0]
        #Need to fit things to the nearest 1m since the hyperspectral data is 1m
        #Centroid is floored instead for consistency with affine transformer
        plot_centroid = math.floor(first_row.easting_plot), math.floor(first_row.northing_plot)
        plot_width = first_row.plotSize ** (1/2)
        min_x, min_y = plot_centroid[0] - (plot_width//2), plot_centroid[1] - (plot_width//2)
        max_x, max_y = min_x + plot_width, min_y + plot_width

        plot_bbox = shapely.box(min_x, min_y, max_x, max_y)

        hs, hs_bands = self.grab_hs(plot_bbox)
        assert hs.shape[0] == plot_width and hs.shape[1] == plot_width, 'hyperspectral plot does not match plot dims'
        chm = self.grab_chm(plot_bbox)
        assert chm.shape[0] == plot_width*10 and chm.shape[1] == plot_width*10, 'chm plot does not match plot dims'
        rgb = self.grab_rgb(plot_bbox)
        assert rgb.shape[0] == plot_width*10 and rgb.shape[1] == plot_width*10, 'rgb plot does not match plot dims'
        ttops = self.grab_ttops(plot_bbox)

        origin = min_x, max_y

        return Plot(
            utm_origin=origin,
            width = plot_width,
            rgb = rgb,
            hyperspectral= hs,
            hyperspectral_bands= hs_bands,
            tree_tops= ttops,
            canopy_height_model= chm,
            potential_trees= selected_plot
        )



    def get_crop_values(self, west_bound, north_bound, scale, tile_bounds):
        affine = AffineTransformer(from_origin(west_bound, north_bound, scale, scale))

        max_y, min_x = affine.rowcol(tile_bounds[0], tile_bounds[1])
        min_y, max_x = affine.rowcol(tile_bounds[2], tile_bounds[3])

        return (min_x, min_y, max_x, max_y)

    #TODO: Filter for selected hyperspectral bands
    #Drop first and last 5 and water bands
    def grab_hs(self, bbox):
        hs_grabs = []
        bounds_list = []
        #bands = None
        tiles = self.__get_relevant_entries__(bbox, self.h5_tiles)
        for ix, tile in tiles.iterrows():
            min_x, min_y, max_x, max_y = self.get_crop_values(tile.file_west_bound, tile.file_north_bound, 1, tile.geometry.bounds)
            bounds_list.append((tile.file_west_bound, tile.file_north_bound))
            hs_file = hp.File(tile.filepath.path, 'r')
            print(tile.filepath.path)


            bands = hs_file[self.sitename]["Reflectance"]["Metadata"]['Spectral_Data']['Wavelength'][:]
            hs_grab = hs_file[self.sitename]["Reflectance"]["Reflectance_Data"][min_y:max_y,min_x:max_x,...]/10000
            hs_grab = hs_grab.astype(np.float32)
            #TODO: Clean up values over/under 1
            hs_grabs.append(hs_grab)
            hs_file.close()

        if len(hs_grabs) == 1:
            return hs_grabs[0], bands
        else:
            return self.__concat_plots__(hs_grabs, bounds_list), bands

    
    def grab_chm(self, bbox) -> np.ndarray:
        chm_grabs = []
        bounds_list = []
        tiles = self.__get_relevant_entries__(bbox, self.chm_tiles)
        for ix, tile in tiles.iterrows():
            min_x, min_y, max_x, max_y = self.get_crop_values(tile.file_west_bound, tile.file_north_bound, .1, tile.geometry.bounds)
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
        ttop_grabs = []
        tiles = self.__get_relevant_entries__(bbox, self.ttop_tiles)
        for ix, tile in tiles.iterrows():
            if tile.filepath.path == self.open_tops_file:
                cur_tops = self.open_tops
            else:
                cur_tops = gpd.read_file(tile.filepath.path)
                self.open_tops = cur_tops
                self.open_tops_file = tile.filepath.path
            cur_tops = cur_tops.clip(tile.geometry)
            ttop_grabs.append(cur_tops)
        
        if len(ttop_grabs) == 1:
            return ttop_grabs[0]
        else:
            return pd.concat(ttop_grabs)

    
    
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



class TreeBuilder:
    def __init__(self, plot: Plot):
        self.plot = plot
        self.tree_segments = self.segment_trees

    def segment_trees(self):
        mask = self.plot.canopy_height_model>2

        #CAN ALSO TRY SCIPY METHODS HERE
        mask = morphology.remove_small_holes(mask)
        mask = morphology.remove_small_objects(mask)

        markers = np.zeros(self.plot.rgb.shape[0:2], dtype=np.int32)
        for ix, rowcol in enumerate(self.plot.tree_tops_local):
            markers[rowcol] = ix

        labelled = watershed(sobel(rgb2gray(self.plot.rgb)), markers=markers, mask=mask, compactness=0.01)
    
        #NEXT STEPS: https://stackoverflow.com/questions/49774179/python-get-boundingbox-coordinates-for-each-cluster-in-segmentation-map-2d-num
        #Do argwhere for each index
        #crop and keep track of bbox?
        pass
    
    def identify_trees(self):
        pass

    


if __name__ == "__main__":

    test = PlotBuilder(
        sitename = "NIWO",
        h5_files= 'W:/Classes/Research/datasets/hs/original/NEON.D13.NIWO.DP3.30006.001.2020-08.basic.20220516T164957Z.RELEASE-2022',
        chm_files= 'C:/Users/tonyt/Documents/Research/datasets/chm/niwo_valid_sites_test',
        ttop_files = "C:/Users/tonyt/Documents/Research/datasets/niwo_tree_tops",
        tree_data_file= 'C:/Users/tonyt/Documents/Research/datasets/tree_locations/NIWO.geojson',
        rgb_files =  r'C:\Users\tonyt\Documents\Research\datasets\rgb\NEON.D13.NIWO.DP3.30010.001.2020-08.basic.20220814T183511Z.RELEASE-2022',
        epsg='EPSG:32613'
    )

    #all_plots = []

    pb = test.build_plots()
    test = next(pb)
    test.find_trees()
    
    print('here')