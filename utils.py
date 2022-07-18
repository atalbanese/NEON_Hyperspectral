from sre_constants import IN
from tkinter import N
import h5_helper as hp
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, IncrementalPCA, FastICA
from skimage.segmentation import slic
from sklearn.neighbors import kneighbors_graph
from skimage.segmentation import mark_boundaries, find_boundaries, watershed
from sklearn import cluster
import os
from tqdm import tqdm
import torch
import torchvision as tv
from multiprocessing import Pool
from pebble import ProcessPool
from concurrent.futures import TimeoutError
from einops import rearrange, reduce
from skimage.color import rgb2hsv
from skimage import exposure
import numpy.ma as ma
import random 
import rasterio as rs

import indexes as ixs


def get_features(inp, feature_band=2):
    return np.reshape(inp, (-1,inp.shape[feature_band]))

def get_norm_stats(inp, **kwargs):
    return None


def pca(data, prep, **kwargs):
    data = get_features(data)
    pc = PCA(**kwargs)
    pc.fit(data)
    data = pc.transform(data)
    if prep:
        data = np.swapaxes(data, 0, 1)
        return data.reshape((kwargs["n_components"], 1000, 1000))
    return data

#Take a dictionary of bands and return a numpy array thats [:,:,3] in red, green, blue order
def make_rgb(band_dict):
    to_stack = [band_dict["red"],
                band_dict["green"],
                band_dict["blue"]]
    return np.stack(to_stack, axis=2)

#Run the han 2018 (https://doi.org/10.3390/app8101883) algorithm for shadow segmentation on rgb/nir bands normalized to 0 to 1 values
def han_2018(to_segment):
    #Get RGB values out of RGB/NIR dictionary
    rgb = make_rgb(to_segment)
    
    #Convert to HSV
    hsv = rgb2hsv(rgb)
    #Get Mixed Property Based Shadow Index (MPSI): (H- I) * (R - NIR)
    #Saturation is equivalent to intensity so using S from HSV
    mpsi = (hsv[:,:,0] - hsv[:,:,1]) * (rgb[:,:,0] - to_segment["nir"])

    #Scale MPSI 0 to 255
    #mpsi = rescale(mpsi, 255)

    #Calculate neighborhood valley threshold from Fan 2011
    #thresh = neighborhood_valley_thresh(mpsi, 3)
    #print(f'found threshold: {thresh}')
    # shadow_mask = mpsi.copy()
    # shadow_mask[shadow_mask<thresh] = 0
    # shadow_mask[shadow_mask>=thresh] = 1

    
    # return mpsi, shadow_mask
    return mpsi #<thresh


def rescale(np_arr, max_val):
    if np_arr.min()<0:
        np_arr += np.abs(np_arr.min())
    return np_arr * max_val

def get_ndvi(to_index):
    return (to_index['nir'] - to_index['red'])/(to_index['nir'] + to_index['red'])



# Neighborhood valley threshold from Fan 2011, implemented based on Han 2018. Finds the threshold with the greatest neighboring variance
# Unlike Otsu threshold, does not depend on a bimodal distribution
def neighborhood_valley_thresh(image: np.ndarray, neighborhood_length: int):
    num_pixels = np.count_nonzero(~np.isnan(image))
    # Calculate neighborhood search radius
    m = int(neighborhood_length//2)
    # Get greyscale histogram
    image_hist, bins = np.histogram(image, bins=255, range=(0, 255))
    # Get probability for each histogram bin - h(g) in han 2018
    h_of_g = image_hist/num_pixels
    potential_thresholds = range(m, image_hist.size-m)

    def h_bar(index):
        return np.nansum(h_of_g[index-m:index+m+1])
    
    def mu_lower(index, p):
        summed = np.nansum(h_of_g[:index+1] * range(0,index+1))
        return summed/p

    def mu_upper(index, p):
        summed = np.nansum(h_of_g[index+1:] * range(index+1,255))
        return summed/p

    best_threshold = -1
    max_var = -1
    for t in potential_thresholds:
        component_1 = (1-h_bar(t))
        p_0 = np.nansum(h_of_g[:t+1])
        mu_0 = mu_lower(t, p_0) ** 2.0
        p_1 = np.nansum(h_of_g[t+1:])
        mu_1 = mu_upper(t, p_1) ** 2.0
        component_2 = (p_0*mu_0) + (p_1 * mu_1)
        var = component_1 * component_2
        if var > max_var:
            max_var = var
            best_threshold = t

    return best_threshold

#TODO: add dask support for speed up/bulk images
def ward_cluster(inp, n_clusters, mask=None, n_neighbors=4):
    print('getting connectivity graph')
    connectivity = kneighbors_graph(
        inp, n_neighbors=n_neighbors, include_self=False
    )

    connectivity = 0.5 * (connectivity + connectivity.T)

    ward = cluster.AgglomerativeClustering(
        n_clusters=n_clusters, linkage="ward", connectivity=connectivity
    )

    print('fitting ward')

    ward.fit(inp)

    return ward.labels_.astype(int)

def save_grid(tb, inp, name, epoch):  
    img_grid = tv.utils.make_grid(inp, normalize=True, scale_each=True)
    tb.add_image(name, img_grid, epoch)

def get_classifications(x):
    norms = torch.nn.functional.softmax(x, dim=1)
    masks = norms.argmax(1).float()

    # masks = torch.unsqueeze(masks, 1) * 255.0/19
    # masks = torch.cat((masks, masks, masks), dim=1)
    return masks

def get_viz_bands():
    return {"red": 654,
            "green": 561, 
            "blue": 482}

def get_shadow_bands():
    return {"red": 654,
            "green": 561, 
            "blue": 482,
            "nir": 825}

def get_landsat_viz():
    return  {"blue": {"lower": 452, "upper": 512},
                     "green": {"lower": 533, "upper": 590},
                     "red": {"lower": 636, "upper": 673}}

def get_bareness_bands():
    return {
        'red': 654,
        "green": 561, 
        "blue": 482,
        'nir': 865,
        'swir': 1610
    }

def get_extra_bands_all():
    return {
        'red': 654,
        "green": 561, 
        "blue": 482,
        'nir': 865,
        'swir': 1610,
        'nitrogen': 1510,
        'lignin': 1754,
        'xanthophyll': 531,
        #These are new per shi 2021 since hard to differentiate spruce and fir
        'disease_water': 1660,
        'red_edge_1': 714,
        'red_edge_2': 733,
        'red_edge_3': 752,
        'swir_2': 2090,
        'swir_3': 2210,
        'swir_4': 2280
    }

def get_extra_bands_scholl():
    return {
        'red': 654,
        "green": 561, 
        "blue": 482,
        'nir': 865,
        'swir': 1610,
        'nitrogen': 1510,
        'lignin': 1754,
        'xanthophyll': 531
    }

def get_extra_bands_shi():
    return {
        'red': 654,
        "green": 561, 
        "blue": 482,
        'nir': 865,
        'swir': 1610,
        'disease_water': 1660,
        'red_edge_1': 714,
        'red_edge_2': 733,
        'red_edge_3': 752,
        'swir_2': 2090,
        'swir_3': 2210,
        'swir_4': 2280
    }



def plot_output_files(dir):
    for file in os.listdir(dir):
        if ".npy" in file:
            y = np.load(os.path.join(dir,file))
            plt.imshow(y)
            plt.show()

def do_pca(args):
    #print(args)
    file, in_dir, out_dir = args
 
    if ".h5" in file:
        new_file = file.split(".")[0] + ".npy"
        if not os.path.exists(os.path.join(out_dir,new_file)):
        
            img = hp.pre_processing(os.path.join(in_dir,file), get_all=True)["bands"]
            img = pca(img, True, n_components=10)
            return (os.path.join(out_dir,new_file), img)

        else:
            return "file already exists"



def img_stats(dir, out_dir, num_channels=30):
    files = os.listdir(dir)
    psum = np.zeros((num_channels,), dtype=np.float64) 
    sq = np.zeros((num_channels,), dtype=np.float64) 
    num_files = 0
    count= 0
    for file in tqdm(files):
        if ".npy" in file:
            num_files += 1
            try:
                img = np.load(os.path.join(dir, file))
                img = rearrange(img, 'h w c -> c (h w)')
                sum = np.nansum(img, axis=1)
                psum += sum
                sq += np.nansum((img**2), axis=1)
                count+= np.count_nonzero(~np.isnan(img))
            except ValueError as e:
                print(e)
                continue
    #count = num_files *1000 * 1000
    total_mean = psum/count
    total_var = (sq/count) - (total_mean**2)
    total_std = np.sqrt(total_var)

    np.save(os.path.join(out_dir, "mean.npy"), total_mean)
    np.save(os.path.join(out_dir, "std.npy"), total_std)

def img_stats_min_max(dir, out_dir, num_channels=10):
    files = os.listdir(dir)
    max = -9999
    min = 9999
    for file in tqdm(files):
        if ".npy" in file:
            try:
                img = np.load(os.path.join(dir, file))
                im_min = img.min()
                im_max = img.max()

                if im_min < min:
                    min = im_min
                
                if im_max > max:
                    max = im_max
            except ValueError as e:
                print(e)
                continue
    #count = num_files *1000 * 1000
    print(f'Min: {min}')
    print(f'Max: {max}')

def img_stats_chm_max(in_dir):
    files = os.listdir(in_dir)
    max = 0
    for file in tqdm(files):
        if ".tif" in file:
            try:
                chm_open = rs.open(os.path.join(in_dir, file))
                img = chm_open.read().astype(np.float32)
                if img.max() > max:
                    max = img.max()
            except ValueError as e:
                print(e)
                continue
    #count = num_files *1000 * 1000
    print(f'max value found: {max}')

def img_stats_chm(in_dir):
    files = os.listdir(in_dir)
    psum = 0
    sq = 0
    count= 0
    for file in tqdm(files):
        if ".tif" in file:
            try:
                chm_open = rs.open(os.path.join(in_dir, file))
                img = chm_open.read().astype(np.float32)
                img[img==-9999] =np.nan
                sum = np.nansum(img)
                psum += sum
                sq += np.nansum((img**2))
                count+= np.count_nonzero(~np.isnan(img))
            except ValueError as e:
                print(e)
                continue
    total_mean = psum/count
    total_var = (sq/count) - (total_mean**2)
    total_std = np.sqrt(total_var)

    print(f'mean: {total_mean}')
    print(f'std: {total_std}')


def save_bands(args):
    file, in_dir, out_dir, bands = args

    if ".h5" in file:
        new_file = file.split(".")[0] + "viz_bands.npy"
        if not os.path.exists(os.path.join(out_dir,new_file)):
            img = hp.pre_processing(os.path.join(in_dir,file), wavelength_ranges=bands)["bands"]
            img = hp.stack_all(img)
            np.save(os.path.join(out_dir,new_file), img)


        else:
            return "file already exists"
    else: 
        return "not an h5 file"

def get_masks(args):
    file, in_dir, out_dir = args
    if ".h5" in file:
        #bare_file = file.split(".")[0] + '_bare_mask.npy'
        shadow_file = file.split(".")[0] + '_sunlit_true_shadow_false.npy'
        if not os.path.exists(os.path.join(out_dir,shadow_file)):
            img = hp.pre_processing(os.path.join(in_dir, file), wavelength_ranges=get_bareness_bands())["bands"]
            b_i = img['red'] + img['swir'] - img['nir']
            bare = b_i > 0.3
            masked_img = {}
            for key, value in img.items():
                value[bare] = np.nan
                masked_img[key] = value
            shadows = han_2018(masked_img)
            
            #np.save(os.path.join(out_dir, bare_file), bare)
            np.save(os.path.join(out_dir, shadow_file), shadows)
            return True

#Saves mask where TRUE = Vegetation
def ndvi_mask(args):
    file, in_dir, out_dir = args
    if ".h5" in file:
        mask_file = file.split(".")[0] + 'ndvi_mask.npy'
        if not os.path.exists(os.path.join(out_dir,mask_file)):
            img = hp.pre_processing(os.path.join(in_dir, file), wavelength_ranges=get_bareness_bands())["bands"]
            ndvi = get_ndvi(img)
            veg = ndvi > 0.5
            np.save(os.path.join(out_dir, mask_file), veg)
            return True



def masked_pca(args):
    file, in_dir, out_dir, mask_dir = args
    if ".h5" in file:
        pca_file = file.split(".")[0] + '_pca.npy'
        #bare_file = file.split(".")[0] + '_bare_mask.npy'
        shadow_file = file.split(".")[0] + 'ndvi_mask.npy'
        if not os.path.exists(os.path.join(out_dir, pca_file)):
            img = hp.pre_processing(os.path.join(in_dir,file), get_all=True)["bands"][:,:,1:]
            #bare_mask = np.load(os.path.join(mask_dir, bare_file))
            shadow_mask = ~np.load(os.path.join(mask_dir, shadow_file))
            img[shadow_mask] = np.nan
            bad_mask = np.zeros_like(shadow_mask)
            for i in range(0, img.shape[-1]):
                z = img[:,:,i]
                y = z>1
                bad_mask += y
            img[bad_mask] = np.nan
            img = rearrange(img, 'h w c -> (h w) c')
            masked = ma.masked_invalid(img)
            mask = masked.mask[:, 0:10]
            to_pca = ma.compress_rows(masked)
            pc = PCA(n_components=10, svd_solver='randomized')
            pc.fit(to_pca)
            data = pc.transform(to_pca)
            out = np.empty(mask.shape, dtype=np.float64)
            np.place(out, mask, np.nan)
            np.place(out, ~mask, data)
            out = rearrange(out, '(h w) c -> h w c', h=1000, w=1000)
            np.save(os.path.join(out_dir, pca_file), out)


            return pca_file

def masked_ica(args):
    file, in_dir, out_dir, mask_dir = args
    if ".h5" in file:
        ica_file = file.split(".")[0] + '_ica_whitened.npy'
        ndvi_file = file.split(".")[0] + 'ndvi_mask.npy'
        if not os.path.exists(os.path.join(out_dir, ica_file)):
            img = hp.pre_processing(os.path.join(in_dir,file), get_all=True)["bands"][:,:,1:]
            #bare_mask = np.load(os.path.join(mask_dir, bare_file))
            shadow_mask = ~np.load(os.path.join(mask_dir, ndvi_file))
            img[shadow_mask] = np.nan
            bad_mask = np.zeros_like(shadow_mask)
            for i in range(0, img.shape[-1]):
                z = img[:,:,i]
                y = z>1
                bad_mask += y
            img[bad_mask] = np.nan
            #img[img>1] = np.nan
            img = rearrange(img, 'h w c -> (h w) c')
            masked = ma.masked_invalid(img)
            mask = masked.mask[:, 0:10]
            to_pca = ma.compress_rows(masked)
            pc = FastICA(n_components=10,
                        random_state=0,
                        tol=1e-3,
                        max_iter=300)
            pc.fit(to_pca)
            data = pc.transform(to_pca)
            out = np.empty(mask.shape, dtype=np.float64)
            np.place(out, mask, np.nan)
            np.place(out, ~mask, data)
            out = rearrange(out, '(h w) c -> h w c', h=1000, w=1000)
            np.save(os.path.join(out_dir, ica_file), out)


            return ica_file



def make_superpixels_viz(args):
    file, in_dir, out_dir = args
    if ".h5" in file:
        super_file = file.split(".")[0] + '_superpixel.npy'
        if not os.path.exists(os.path.join(out_dir, super_file)):
            rgb = hp.pre_processing(os.path.join(in_dir, file), wavelength_ranges=get_viz_bands())
            rgb = hp.make_rgb(rgb["bands"])
            mask = rgb == rgb
            rgb[~mask] = 0
            segments = slic(rgb, n_segments=10000, slic_zero=True, mask=mask[:,:,0])
            # rgb = exposure.adjust_gamma(rgb, 0.5) 
            # plt.imshow(mark_boundaries(rgb, segments))
            # plt.show()
            np.save(os.path.join(out_dir, super_file), segments)

def make_superpixels_chm(args):
    file, in_dir, out_dir = args
    if ".tif" in file:
        chm_file = file.split(".")[0] + '_superpixel.npy'
        # rgb_file =  file.split(".")[0].replace('CHM', 'reflectance') + '.h5'
        if not os.path.exists(os.path.join(out_dir, chm_file)):
            # rgb = hp.pre_processing(os.path.join(orig_dir, rgb_file), wavelength_ranges=get_viz_bands())
            # rgb = hp.make_rgb(rgb["bands"])

            chm_open = rs.open(os.path.join(in_dir, file))
            chm = chm_open.read().astype(np.float32)
            chm[chm==-9999] = np.nan
            chm = chm.squeeze(axis=0)
            mask = chm != 0
            chm[~mask] = 0
            c_max = chm.max()
            chm = chm/c_max
            # plt.imshow(chm)
            # plt.show()
            # plt.imshow(rgb)
            # plt.show()
            segments = slic(chm, n_segments=mask.sum()//36, slic_zero=True, mask=mask)
            #segments = watershed(chm, markers=mask.sum()//36, compactness=0.01, mask=mask)
           
            np.save(os.path.join(out_dir, chm_file), segments)

def make_superpixels_masked(args):
    file, in_dir, out_dir, mask_dir = args
    if ".h5" in file:
        super_file = file.split(".")[0] + '_superpixel.npy'
        mask_file = file.split(".")[0] + '_pca.npy'
        if not os.path.exists(os.path.join(out_dir, super_file)):
            rgb = hp.pre_processing(os.path.join(in_dir, file), wavelength_ranges=get_viz_bands())
            rgb = hp.make_rgb(rgb["bands"])
            mask = np.load(os.path.join(mask_dir, mask_file))
            mask = mask == mask
            mask = mask[:,:, 0]
            rgb[~mask] = 0
            

            segments = slic(rgb, n_segments=mask.sum()//64, slic_zero=True, mask=mask)
            # rgb = exposure.adjust_gamma(rgb, 0.5) 
            # plt.imshow(mark_boundaries(rgb, segments, mode='subpixel'))
            # plt.show()
            np.save(os.path.join(out_dir, super_file), segments)

def bulk_shadow_index(args):
    file, in_dir, out_dir = args
    if ".h5" in file:
        out_file = file.split('.')[0] + '_mpsi.npy'
        if not os.path.exists(os.path.join(out_dir, out_file)):
            img = hp.pre_processing(os.path.join(in_dir, file), wavelength_ranges=get_shadow_bands())['bands']
            mpsi = han_2018(img)
            np.save(os.path.join(out_dir, out_file), mpsi)

def select_extra_bands(args):
    file, in_dir, out_dir, fn = args
    if ".h5" in file:
        out_file = file.split('.')[0] + '_extrabands.npy'
        if not os.path.exists(os.path.join(out_dir, out_file)):
            img = hp.pre_processing(os.path.join(in_dir, file), wavelength_ranges=fn())['bands']
            bands = hp.stack_all(img, axis=2)
            np.save(os.path.join(out_dir, out_file), bands)

def get_all_indexes(args):
    file, in_dir, out_dir, mask_dir = args
    if ".h5" in file:
        out_file = file.split('.')[0] + '_indexbands.npy'
        mask_file = file.split(".")[0] + '_pca.npy'
        #

        if not os.path.exists(os.path.join(out_dir, out_file)):
            mask = np.load(os.path.join(mask_dir, mask_file))
            mask = mask == mask
            mask = reduce(mask, 'h w c -> h w ()', 'max').squeeze()
            img = hp.pre_processing(os.path.join(in_dir, file), wavelength_ranges=ixs.BANDS)['bands']
            masked_img={}
            for key, band in img.items():
                band[~mask] =np.nan
                band[band == 0] = np.nan
                masked_img[key] = band
            to_stack = []
            for ix_fn in ixs.INDEX_FNS:
                to_append = ix_fn(masked_img)
                to_append[to_append == np.inf] = np.nan
                to_append[to_append == -np.inf] = np.nan


                to_stack.append(to_append)
            
            out = np.stack(to_stack)
            np.save(os.path.join(out_dir, out_file), out)
    
    

def build_inc_pca(args):
    file, in_dir, pca_solver = args
    if ".h5" in file:
        img = hp.pre_processing(os.path.join(in_dir,file), get_all=True)["bands"][:,:,1:]

        img = rearrange(img, 'h w c -> (h w) c')
        masked = ma.masked_invalid(img)
        #mask = masked.mask[:, 0:30]
        to_pca = ma.compress_rows(masked)
        pca_solver.partial_fit(to_pca)


    return pca_solver

def do_inc_pca(args):
    file, in_dir, out_dir, pca_solver = args
    if ".h5" in file:
        pca_file = file.split(".")[0] + '_pca.npy'
        if not os.path.exists(os.path.join(out_dir, pca_file)):
            img = hp.pre_processing(os.path.join(in_dir,file), get_all=True)["bands"][:,:,1:]
            #bare_mask = np.load(os.path.join(mask_dir, bare_file))
            img = rearrange(img, 'h w c -> (h w) c')
            masked = ma.masked_invalid(img)
            mask = masked.mask[:, 0:10]
            to_pca = ma.compress_rows(masked)
            data = pca_solver.transform(to_pca)
            out = np.empty(mask.shape, dtype=np.float64)
            np.place(out, mask, np.nan)
            np.place(out, ~mask, data)
            out = rearrange(out, '(h w) c -> h w c', h=1000, w=1000)
            np.save(os.path.join(out_dir, pca_file), out)

            return pca_file


def save_solar_stats(args):
    file, in_dir, out_dir = args
    if ".h5" in file:
        sol_file = file.split(".")[0] + '_solar.npy'
        if not os.path.exists(os.path.join(out_dir, sol_file)):
            img = hp.pre_processing(os.path.join(in_dir,file), meta_only=True)["meta"]
            np.save(os.path.join(out_dir, sol_file), img['azimuth'])
            return sol_file






def bulk_process(pool, dirs, fn, **kwargs):
    files = os.listdir(dirs[0])
    dirs = [[folder] * len(files) for folder in dirs]

    args_list = list(zip(files, *dirs))

    future = pool.map(fn, args_list, timeout=1000)
    iterator = future.result()
    while True:
        try:
            n = next(iterator)
            if isinstance(n, tuple):
                print(n[0])
                np.save(*n)
                #print(n[0])
            else:
                print(n)
        except TimeoutError as e:
            print(e.args)
            continue
        except ValueError as e:
            print(e)
            continue
        except FileNotFoundError as e:
            print(e)
            continue
        except StopIteration:
            break





if __name__ == '__main__':

    # with ProcessPool(4) as pool:
    #     IN_DIR = '/data/shared/src/aalbanese/datasets/hs/NEON_refl-surf-dir-ortho-mosaic/NEON.D01.HARV.DP3.30006.001.2019-08.basic.20220407T001553Z.RELEASE-2022'
    #     OUT_DIR = '/data/shared/src/aalbanese/datasets/hs/masks/HARV'
    #     bulk_process(pool, IN_DIR, OUT_DIR, get_masks)

    IN_DIR = 'W:/Classes/Research/datasets/hs/original/NEON.D13.NIWO.DP3.30006.001.2020-08.basic.20220516T164957Z.RELEASE-2022'
    MASK_DIR = 'C:/Users/tonyt/Documents/Research/datasets/ndvi/harv'
    BANDS = get_shadow_bands()

    

    FILE = 'NEON_D13_NIWO_DP3_445000_4432000_reflectance.h5'
    OUT_DIR = 'C:/Users/tonyt/Documents/Research/datasets/indexes/niwo/'
    ICA_DIR = 'C:/Users/tonyt/Documents/Research/datasets/ica/niwo_10_channels'
    PCA_DIR = 'C:/Users/tonyt/Documents/Research/datasets/pca/niwo_masked_10'
    FN = get_extra_bands_all

    chm_fold = 'C:/Users/tonyt/Documents/Research/datasets/chm/niwo'

    #make_superpixels_chm((FILE, chm_fold, OUT_DIR, IN_DIR))

    # test = hp.pre_processing(os.path.join(IN_DIR, FILE), get_all=True)['bands']
    # for i in range(0, test.shape[2]):
    #     plt.imshow(test[:,:,i])
    #     plt.show()

    #masked_pca((FILE, IN_DIR, 'test', MASK_DIR))


    # ndvi_mask((FILE, IN_DIR, IN_DIR))

    #masked_ica((FILE, IN_DIR, IN_DIR, MASK_DIR))

    # make_superpixels_viz((FILE, IN_DIR, 'test'))


    #img_stats_chm(chm_fold)
    #img_stats_min_max('C:/Users/tonyt/Documents/Research/datasets/pca/harv_2022_10_channels', '')

    #get_all_indexes((FILE, IN_DIR, OUT_DIR))
    

    with ProcessPool(4) as pool:
        bulk_process(pool, [IN_DIR, OUT_DIR, PCA_DIR], get_all_indexes)

    # # # with ProcessPool(4) as pool:
    # # #     bulk_process(pool, [IN_DIR, ICA_DIR, MASK_DIR], masked_ica)

    # with ProcessPool(4) as pool:
    #     bulk_process(pool, [IN_DIR, PCA_DIR, MASK_DIR], masked_pca)

    # with ProcessPool(4) as pool:
    #     bulk_process(pool, [chm_fold, OUT_DIR], make_superpixels_chm)

    #get_masks((IMG, IMG_DIR, MASK_DIR))

    #img_stats('C:/Users/tonyt/Documents/Research/datasets/extracted_plots/harv_2022/plots_pca', 'C:/Users/tonyt/Documents/Research/datasets/extracted_plots/harv_2022/plots_pca/stats', num_channels=10)

    # with ProcessPool(4) as pool:
    #     bulk_process(pool, [IN_DIR, MASK_DIR], get_masks)



    # PCA_SOLVER = IncrementalPCA(n_components=10)

    # for f in tqdm(random.sample(os.listdir(IN_DIR), 40)):
    #     PCA_SOLVER = build_inc_pca((f, IN_DIR, PCA_SOLVER))

    # with ProcessPool(4) as pool:
    #     bulk_process(pool, [IN_DIR, OUT_DIR, PCA_SOLVER], do_inc_pca)

    # img_stats(OUT_DIR, os.path.join(OUT_DIR, 'stats'), num_channels=10)

