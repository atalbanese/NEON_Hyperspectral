import annotation
import matplotlib.pyplot as plt
import rasterio as rs
import h5py as hp
import numpy as np
from einops import rearrange
#Side by side RGB and HS and CHM
def fig_3():
   pb = annotation.PlotBuilder(
    sitename = "NIWO",
    h5_files= 'W:/Classes/Research/datasets/hs/original/NEON.D13.NIWO.DP3.30006.001.2020-08.basic.20220516T164957Z.RELEASE-2022',
    chm_files= 'C:/Users/tonyt/Documents/Research/datasets/chm/niwo_valid_sites_test',
    ttop_files = "C:/Users/tonyt/Documents/Research/datasets/niwo_tree_tops",
    tree_data_file= 'C:/Users/tonyt/Documents/Research/datasets/tree_locations/NIWO_by_me.geojson',
    rgb_files =  r'C:\Users\tonyt\Documents\Research\datasets\rgb\NEON.D13.NIWO.DP3.30010.001.2020-08.basic.20220814T183511Z.RELEASE-2022',
    epsg='EPSG:32613',
    base_dir=r'C:\Users\tonyt\Documents\Research\thesis_final'
   )

   for plot in pb.build_plots():
       plot.plot_me()

def fig_8():
    pb = annotation.PlotBuilder(
    sitename = "NIWO",
    h5_files= 'W:/Classes/Research/datasets/hs/original/NEON.D13.NIWO.DP3.30006.001.2020-08.basic.20220516T164957Z.RELEASE-2022',
    chm_files= 'C:/Users/tonyt/Documents/Research/datasets/chm/niwo_valid_sites_test',
    ttop_files = "C:/Users/tonyt/Documents/Research/datasets/niwo_tree_tops",
    tree_data_file= 'C:/Users/tonyt/Documents/Research/datasets/tree_locations/NIWO_by_me.geojson',
    rgb_files =  r'C:\Users\tonyt\Documents\Research\datasets\rgb\NEON.D13.NIWO.DP3.30010.001.2020-08.basic.20220814T183511Z.RELEASE-2022',
    epsg='EPSG:32613',
    base_dir=r'C:\Users\tonyt\Documents\Research\thesis_final',
    plot_hs_dif=True

   )
    
    for plot in pb.build_plots():
        pass
        

if __name__ == "__main__":
    fig_8()