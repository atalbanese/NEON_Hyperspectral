import base_classes
import os
import numpy as np
import geopandas as gpd
import pandas as pd


class PlotBuilder:
    """Takes in known data for a study site and returns Plot objects populated with Tree objects, where each Tree has been paired with an identified crown
        All plots should be contained within a Study Site object. The goal is total abstraction"""
    def __init__(
        self,
        sitename: str,
        h5_files: str,
        chm_files: str,
        ttop_files: str,
        tree_data_file: str
        ):
        #Static Vars
        self.sitename = sitename
        self.h5_files = os.scandir(h5_files)
        self.chm_files = os.scandir(chm_files)
        self.ttop_files = os.scandir(ttop_files)
        self.plot_data_file = gpd.read_file(tree_data_file)
        self.all_plot_ids = list(np.unique(self.plot_data_file.plotID))

        #Mutable Vars
        self.current_plot_id = self.all_plot_ids[0]


    


if __name__ == "__main__":

    test = PlotBuilder(
        sitename = "NIWO",
        h5_files= 'W:/Classes/Research/datasets/hs/original/NEON.D13.NIWO.DP3.30006.001.2020-08.basic.20220516T164957Z.RELEASE-2022',
        chm_files= 'C:/Users/tonyt/Documents/Research/datasets/chm/niwo_valid_sites_test',
        ttop_files = 'C:/Users/tonyt/Documents/Research/datasets/lidar/niwo_point_cloud/valid_sites_ttops',
        tree_data_file= 'C:/Users/tonyt/Documents/Research/datasets/tree_locations/NIWO.geojson'
    )

    print('here')