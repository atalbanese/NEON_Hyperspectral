import os
import numpy as np
import geopandas as gpd
import pandas as pd


def reconcile_tree_data(
    curated_trees_file: str,
    plots_file: str,
    site_name: str,
    save_loc: str,
    epsg: str = 'EPSG:32613',
    filter_species: list = ['DAFRF', 'SALIX']
    ):

    curated = pd.read_csv(curated_trees_file, usecols=['plotID', 'individualID', 'adjEasting', 'adjNorthing', 'taxonID', 'height'])
    curated = curated.rename(columns={'adjEasting': 'easting',
                                          'adjNorthing': 'northing'})

    #Only select data for which we have coordinates
    curated = curated.dropna()

    #Remove any specified taxa, usually those for which there are very few examples
    if filter_species is not None:
        for f in filter_species:
            curated = curated.loc[curated['taxonID'] != f]
        

   #Now we have all the trees we want, we need to get data on the individual plots.

    plots = pd.read_csv(plots_file, usecols=['plotID', 'siteID', 'subtype', 'easting', 'northing', 'plotSize'])
    plots = plots.loc[plots['siteID'] == site_name]
    plots = plots.loc[plots['subtype'] == 'basePlot']

    combined = curated.merge(plots, how='left', on='plotID')

    combined = combined.rename(columns={
                                    'easting_x': 'easting_tree',
                                    'northing_x': 'northing_tree',
                                    'easting_y': 'easting_plot',
                                    'northing_y': 'northing_plot'
                                })
    data_gdf = gpd.GeoDataFrame(combined, geometry=gpd.points_from_xy(curated.easting, curated.northing, curated.height), crs=epsg)
    data_gdf.to_file(save_loc)

if __name__ == "__main__":

    reconcile_tree_data(
        curated_trees_file="W:/Classes/Research/neon_niwo_mapped_struct_de_dupe.csv",
        plots_file='W:/Classes/Research/All_NEON_TOS_Plots_V8/All_NEON_TOS_Plots_V8/All_NEON_TOS_Plot_Centroids_V8.csv',
        site_name='NIWO',
        save_loc='C:/Users/tonyt/Documents/Research/datasets/tree_locations/NIWO.geojson'
    )

    