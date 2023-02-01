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
        potential_trees: gpd.GeoDataFrame,
        epsg: str,
        base_dir: str,
        name: str,
        sitename: str
    ):
        self.name = name
        self.sitename = sitename
        self.base_dir = base_dir
        self.epsg = epsg
        self.width = width
        self.utm_origin = utm_origin
        self.rgb = rgb
        self.hyperspectral = hyperspectral
        self.hyperspectral_bands = hyperspectral_bands
        self.tree_tops = tree_tops.reset_index(drop=True)
        self.canopy_height_model = canopy_height_model
        self.potential_trees = potential_trees
        self.cm_affine = AffineTransformer(from_origin(self.utm_origin[0], self.utm_origin[1], .1, .1))
        self.m_affine = AffineTransformer(from_origin(self.utm_origin[0], self.utm_origin[1], 1, 1))
        self.tree_tops_local = self.make_local_tree_tops()

        self.identified_trees = list()
    
    def make_local_tree_tops(self):
        out_list = []
        for ix, tree in self.tree_tops.iterrows():
            out_list.append(self.cm_affine.rowcol(tree.geometry.x, tree.geometry.y))
        return out_list

    def drop_ttops(self, include_idxs):
        self.tree_tops = self.tree_tops.iloc[include_idxs]

    def find_trees(self):
        #This will modifiy some of the data in this object as well. They are intertwined like the forest and the sky
        tree_builder = TreeBuilder(self)
        self.identified_trees = tree_builder.build_trees()

    def plot_and_check_trees(self, save_size = 8):
        for tree in self.identified_trees:
            tp = TreePlotter(tree)

class Tree:
    def __init__(
        self,
        hyperspectral: np.ndarray,
        rgb: np.ndarray,
        rgb_mask: np.ndarray,
        hyperspectral_bands: np.ndarray,
        chm: np.ndarray,
        site_id: str,
        plot_id: str,
        utm_origin: tuple,
        individual_id:str,
        taxa: str,
        plot: Plot,
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
    
    def go_back_to_old_mask(self):
        self.rgb_mask = self.old_rgb_mask
        self.hyperspectral_mask = self.make_hs_mask()

    #TODO: Will Be switched to LoadedTree
    # def get_masked_hs(self, out_type ='numpy'):
    #     if out_type == 'numpy':
    #         return self.hyperspectral[self.hyperspectral_mask]
    #     if out_type == 'torch':
    #         return torch.from_numpy(self.hyperspectral[self.hyperspectral_mask])


    # def get_masked_rgb(self, out_type ='numpy'):
    #     if out_type == 'numpy':
    #         return self.rgb[self.rgb]
    #     if out_type == 'torch':
    #         return torch.from_numpy(self.rgb[self.rgb_mask])
    
    def save(self):
        savedir = os.path.join(self.plot.base_dir, self.plot.sitename, self.plot.name)
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
            site_id = self.site_id
            )


    


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
        epsg: str,
        base_dir: str,
        ):
        #Static Vars
        self.base_dir = base_dir
        self.sitename = sitename
        self.epsg = epsg
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
            potential_trees= selected_plot,
            epsg = self.epsg,
            base_dir= self.base_dir,
            name = self.current_plot_id,
            sitename=self.sitename
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
        #self.labelled_index = original_index
        self.ttop_index = original_index - 1
        #Origin = upper left, row-col
        #Anything local is y-x
        #Anything utm is x-y
        #If only there was some way to standardize this!
        self.local_origin = self.y_min, self.x_min

        self.utm_origin = affine.xy(*self.local_origin)
        self.utm_x_min, self.utm_y_min = affine.xy(self.y_min, self.x_min)
        self.utm_x_max, self.utm_y_max = affine.xy(self.y_max, self.x_max)
        # self.height = self.y_max - self.y_min
        # self.width = self.x_max - self.x_min

        # self.local_centroid = self.y_min + (self.height//2), self.x_min + (self.width//2)
        # self.utm_centroid = affine.xy(*self.local_centroid)

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


class TreeBuilder:
    def __init__(self, plot: Plot):
        self.plot = plot
        #Find visible canopies using watershed segmentation + lidar treetops locations
        self.canopy_segments, self.labelled_plot = self.segment_canopy()
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

            #Potentially do this to clean up labels
            #label_mask = morphology.remove_small_objects(morphology.erosion(label_mask), min_size=225)
            #TODO: make this a button on the tree approver

            #Need to get HS and RGB onto same grid to get HS mask. This could all maybe be moved to Tree?

            hs_x_min, hs_y_min = math.floor(local_x_min/10), math.floor(local_y_min/10)
            hs_x_max, hs_y_max = math.ceil(local_x_max/10), math.ceil(local_y_max/10)

            x_left_pad, x_right_pad = local_x_min-(hs_x_min*10), (hs_x_max*10) - local_x_max
            y_up_pad, y_down_pad = local_y_min - (hs_y_min*10), (hs_y_max*10) - local_y_max 

            rgb_mask = np.pad(label_mask, ((y_up_pad, y_down_pad), (x_left_pad, x_right_pad)))
            chm = np.pad(chm, ((y_up_pad, y_down_pad), (x_left_pad, x_right_pad)))
            rgb = np.pad(rgb, ((y_up_pad, y_down_pad), (x_left_pad, x_right_pad), (0,0)))

            hs = self.plot.hyperspectral[hs_y_min:hs_y_max, hs_x_min:hs_x_max, ...]
            #TODO: check if hs bands really are identical between all datasets
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
                plot=self.plot
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
                    else:
                        #If there are no treetops within distance just throw this one out
                        tree_skip_list.add(ix)
                    #If they are both each others best pair, add them to the pairs list and remove from consideration
                    if best_tree_idx == ix:
                        selected_crowns.add(best_crown_idx)
                        tree_skip_list.add(best_tree_idx)
                        labelled_pairs[int(best_tree_idx)] = int(best_crown_idx)
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

        self.fig, self.axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
        self.hs_ax = self.axes[0]
        self.rgb_ax = self.axes[1]

        self.hs_ax.set_title('1m Hyperspectral Mask')
        self.rgb_ax.set_title('10cm RGB')

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
        self.update()

    def update(self):
        self.hs_ax.clear()
        self.rgb_ax.clear()
        self.draw_hs()
        self.draw_rgb()
        self.hs_im.axes.figure.canvas.draw()
        self.rgb_im.axes.figure.canvas.draw()

    def draw_hs(self):
        # rgb = [700, 546.1, 435.8]
        # rgb_idxs = [self.find_nearest(wave) for wave in rgb]
        
        hs_im = self.hs_ax.imshow(self.tree.hyperspectral_mask, picker=True)
        return hs_im

    def draw_rgb(self):
        
        rgb_im = self.rgb_ax.imshow(mark_boundaries(self.tree.rgb, self.tree.rgb_mask))

        return rgb_im
    
    def find_nearest(self, search_val):
        diff_arr = np.absolute(self.tree.hyperspectral_bands-search_val)
        return diff_arr.argmin()

    def handle_hs_click(self, event):
        x_loc = round(event.mouseevent.xdata)
        y_loc = round(event.mouseevent.ydata)
        print(y_loc)
        self.tree.hyperspectral_mask[y_loc, x_loc] = ~self.tree.hyperspectral_mask[y_loc, x_loc]
    
def annotate_all(**kwargs):
    pb = PlotBuilder(**kwargs)
    for plot in pb.build_plots():
        print(plot.name)
        plot.find_trees()
        plot.plot_and_check_trees()

    


if __name__ == "__main__":

    annotate_all(
        sitename = "NIWO",
        h5_files= 'W:/Classes/Research/datasets/hs/original/NEON.D13.NIWO.DP3.30006.001.2020-08.basic.20220516T164957Z.RELEASE-2022',
        chm_files= 'C:/Users/tonyt/Documents/Research/datasets/chm/niwo_valid_sites_test',
        ttop_files = "C:/Users/tonyt/Documents/Research/datasets/niwo_tree_tops",
        tree_data_file= 'C:/Users/tonyt/Documents/Research/datasets/tree_locations/NIWO.geojson',
        rgb_files =  r'C:\Users\tonyt\Documents\Research\datasets\rgb\NEON.D13.NIWO.DP3.30010.001.2020-08.basic.20220814T183511Z.RELEASE-2022',
        epsg='EPSG:32613',
        base_dir=r'C:\Users\tonyt\Documents\Research\thesis_final'
        )

    # test = PlotBuilder(
    #     sitename = "NIWO",
    #     h5_files= 'W:/Classes/Research/datasets/hs/original/NEON.D13.NIWO.DP3.30006.001.2020-08.basic.20220516T164957Z.RELEASE-2022',
    #     chm_files= 'C:/Users/tonyt/Documents/Research/datasets/chm/niwo_valid_sites_test',
    #     ttop_files = "C:/Users/tonyt/Documents/Research/datasets/niwo_tree_tops",
    #     tree_data_file= 'C:/Users/tonyt/Documents/Research/datasets/tree_locations/NIWO.geojson',
    #     rgb_files =  r'C:\Users\tonyt\Documents\Research\datasets\rgb\NEON.D13.NIWO.DP3.30010.001.2020-08.basic.20220814T183511Z.RELEASE-2022',
    #     epsg='EPSG:32613',
    #     base_dir=r'C:\Users\tonyt\Documents\Research\thesis_final'
    # )

    # #all_plots = []

    # pb = test.build_plots()
    # test = next(pb)
    # test.find_trees()
    # test.plot_and_check_trees()
    
    # print('here')