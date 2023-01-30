import numpy as np
import torch
import geopandas as gpd
import os
import shapely

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

        self.identified_trees = list()

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




class StudyArea:
    pass

if __name__ == "__main__":
    test = TileSet('W:/Classes/Research/datasets/hs/original/NEON.D13.NIWO.DP3.30006.001.2020-08.basic.20220516T164957Z.RELEASE-2022','EPSG:32613','.h5', (-3, -2), 1000)

