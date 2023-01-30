import numpy as np
import torch
import geopandas as gpd




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
        tile_origin: tuple,
        width: int,
        utm_origin: tuple,
        rgb: np.ndarray,
        hyperspectral: np.ndarray,
        hyperspectral_bands: list,
        tree_tops: gpd.GeoDataFrame,
        canopy_height_model: np.ndarray,
        potential_trees: list
    ):
        self.tile_origin = tile_origin
        self.width = width
        self.utm_origin = utm_origin
        self.rgb = rgb
        self.hyperspectral = hyperspectral,
        self.hyperspectral_bands = hyperspectral_bands,
        self.tree_tops = tree_tops,
        self.canopy_height_model = canopy_height_model,
        self.potential_trees = potential_trees

        self.identified_trees = list()

    pass


class StudyArea:
    pass