import base_classes
import os
import numpy as np
import geopandas as gpd
import pandas as pd
import shapely


class PlotBuilder:
    """Takes in known data for a study site and returns Plot objects populated with Tree objects, where each Tree has been paired with an identified crown
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
        self.h5_tiles = base_classes.TileSet(h5_files, epsg, '.h5', (-3,-2), 1000)
        self.chm_tiles = base_classes.TileSet(chm_files, epsg, '.tif', (-3, -2), 1000)
        self.ttop_tiles = base_classes.TileSet(ttop_files, epsg, '.geojson', (-3, -2), 1000)
        self.rgb_tiles = base_classes.TileSet(rgb_files, epsg, '.tif', (-3, -2), 1000)
        self.plot_data_file = gpd.read_file(tree_data_file)
        self.all_plot_ids = list(np.unique(self.plot_data_file.plotID))

        #Will need two affine transforms for 1m and 10cm raster data

        #Mutable Vars
        self.current_plot_id = None

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
        plot_centroid = round(first_row.easting_plot), round(first_row.northing_plot)
        plot_width = first_row.plotSize ** (1/2)
        min_x, min_y = plot_centroid[0] - (plot_width//2), plot_centroid[1] - (plot_width//2)
        max_x, max_y = min_x + plot_width, min_y + plot_width

        plot_bbox = shapely.box(min_x, min_y, max_x, max_y)




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

    print('here')