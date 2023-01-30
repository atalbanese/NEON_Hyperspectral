import tree_classes
import os
import numpy as np
import geopandas as gpd
import pandas as pd


class PlotBuilder:
    def __init__(
        self,
        sitename: str,
        h5_files: str,
        chm_files: str,
        ttop_files: str,
        plot_data_file: str
        ):
        self.sitename = sitename
        self.h5_files = os.list_dir(h5_files)
        self.chm_files = os.list_dir(chm_files)
        self.ttop_files = os.list_dir(ttop_files)
        self.plot_data_file = pd.read_csv(plot_data_file)

        