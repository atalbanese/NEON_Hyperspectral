from multiprocessing.sharedctypes import Value
import h5_helper as hp
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.neighbors import kneighbors_graph
from sklearn import cluster
import os
from tqdm import tqdm
import torch
import torchvision as tv
from multiprocessing import Pool
from pebble import ProcessPool
from concurrent.futures import TimeoutError
from einops import rearrange
from skimage.color import rgb2hsv
from skimage import exposure
import numpy.ma as ma
import random 

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
    mpsi = rescale(mpsi, 255)

    #Calculate neighborhood valley threshold from Fan 2011
    thresh = neighborhood_valley_thresh(mpsi, 3)
    print(f'found threshold: {thresh}')
    # shadow_mask = mpsi.copy()
    # shadow_mask[shadow_mask<thresh] = 0
    # shadow_mask[shadow_mask>=thresh] = 1

    
    # return mpsi, shadow_mask
    return mpsi<thresh


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
            "nir": 865}

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

def plot_output_files(dir):
    for file in os.listdir(dir):
        if ".npy" in file:
            y = np.load(os.path.join(dir,file))
            plt.imshow(y)
            plt.show()

def do_pca(args):
    band = {'b1': 500}
    #print(args)
    file, in_dir, out_dir = args
 
    if ".h5" in file:
        new_file = file.split(".")[0] + ".npy"
        if not os.path.exists(os.path.join(out_dir,new_file)):
            img = hp.pre_processing(os.path.join(in_dir,file), wavelength_ranges=band)
            if np.isnan(img["bands"]['b1']).sum() == 0:
                img = hp.pre_processing(os.path.join(in_dir,file), get_all=True)["bands"]
                img = pca(img, True, n_components=30)
                return (os.path.join(out_dir,new_file), img)
                #return new_file
            else:
                return "nans in file, skipping"
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


def masked_pca(args):
    file, in_dir, out_dir, mask_dir = args
    if ".h5" in file:
        pca_file = file.split(".")[0] + '_pca.npy'
        #bare_file = file.split(".")[0] + '_bare_mask.npy'
        shadow_file = file.split(".")[0] + '_shadow_mask.npy'
        if not os.path.exists(os.path.join(out_dir, pca_file)):
            img = hp.pre_processing(os.path.join(in_dir,file), get_all=True)["bands"]
            #bare_mask = np.load(os.path.join(mask_dir, bare_file))
            shadow_mask = ~np.load(os.path.join(mask_dir, shadow_file))
            img[shadow_mask] = np.nan
            img = rearrange(img, 'h w c -> (h w) c')
            masked = ma.masked_invalid(img)
            mask = masked.mask[:, 0:30]
            to_pca = ma.compress_rows(masked)
            pc = PCA(n_components=30, svd_solver='randomized')
            pc.fit(to_pca)
            data = pc.transform(to_pca)
            out = np.empty(mask.shape, dtype=np.float64)
            np.place(out, mask, np.nan)
            np.place(out, ~mask, data)
            out = rearrange(out, '(h w) c -> h w c', h=1000, w=1000)
            np.save(os.path.join(out_dir, pca_file), out)


            return pca_file

def build_inc_pca(args):
    file, in_dir, pca_solver = args
    if ".h5" in file:
        img = hp.pre_processing(os.path.join(in_dir,file), get_all=True)["bands"]

        img = rearrange(img, 'h w c -> (h w) c')
        masked = ma.masked_invalid(img)
        mask = masked.mask[:, 0:30]
        to_pca = ma.compress_rows(masked)
        pca_solver.partial_fit(to_pca)


    return pca_solver

def do_inc_pca(args):
    file, in_dir, out_dir, pca_solver = args
    if ".h5" in file:
        pca_file = file.split(".")[0] + '_pca.npy'
        if not os.path.exists(os.path.join(out_dir, pca_file)):
            img = hp.pre_processing(os.path.join(in_dir,file), get_all=True)["bands"]
            #bare_mask = np.load(os.path.join(mask_dir, bare_file))
            img = rearrange(img, 'h w c -> (h w) c')
            masked = ma.masked_invalid(img)
            mask = masked.mask[:, 0:30]
            to_pca = ma.compress_rows(masked)
            data = pca_solver.transform(to_pca)
            out = np.empty(mask.shape, dtype=np.float64)
            np.place(out, mask, np.nan)
            np.place(out, ~mask, data)
            out = rearrange(out, '(h w) c -> h w c', h=1000, w=1000)
            np.save(os.path.join(out_dir, pca_file), out)

            return pca_file






def bulk_process(pool, dirs, fn, **kwargs):
    files = os.listdir(dirs[0])
    dirs = [[folder] * len(files) for folder in dirs]

    args_list = list(zip(files, *dirs))

    future = pool.map(fn, args_list, timeout=99)
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
        except StopIteration:
            break





if __name__ == '__main__':

    # with ProcessPool(4) as pool:
    #     IN_DIR = '/data/shared/src/aalbanese/datasets/hs/NEON_refl-surf-dir-ortho-mosaic/NEON.D01.HARV.DP3.30006.001.2019-08.basic.20220407T001553Z.RELEASE-2022'
    #     OUT_DIR = '/data/shared/src/aalbanese/datasets/hs/masks/HARV'
    #     bulk_process(pool, IN_DIR, OUT_DIR, get_masks)

    IN_DIR = 'W:/Classes/Research/datasets/hs/original/NEON.D01.HARV.DP3.30006.001.2019-08.basic.20220501T135554Z.RELEASE-2022'
    MASK_DIR = 'C:/Users/tonyt/Documents/Research/datasets/masks/harv_2022'
    BANDS = get_shadow_bands()

    IMG_DIR = 'W:/Classes/Research/datasets/hs/original/NEON.D01.HARV.DP3.30006.001.2019-08.basic.20220501T135554Z.RELEASE-2022'
    IMG= os.path.join(IMG_DIR, 'NEON_D01_HARV_DP3_736000_4703000_reflectance.h5')


    OUT_DIR = 'W:/Classes/Research/datasets/hs/pca/harv_2022'

    get_masks((IMG, IMG_DIR, MASK_DIR))

    # with ProcessPool(4) as pool:
    #     bulk_process(pool, [IN_DIR, MASK_DIR], get_masks)



    #PCA_SOLVER = IncrementalPCA(n_components=30)

    # for f in tqdm(random.sample(os.listdir(IN_DIR), 40)):
    #     PCA_SOLVER = build_inc_pca((f, IN_DIR, PCA_SOLVER))

    # # print(PCA_SOLVER.mean_)
    # # print(PCA_SOLVER.var_)

    # with ProcessPool(4) as pool:
    #     bulk_process(pool, [IN_DIR, OUT_DIR, PCA_SOLVER], do_inc_pca)


    # with ProcessPool(4) as pool:
    #     bulk_process(pool, [IN_DIR, OUT_DIR, BANDS], save_bands)

    #img_stats(OUT_DIR, 'W:/Classes/Research/datasets/hs/pca/harv_2022/stats', num_channels=30)


    #get_bareness_mask((IMG, '/data/shared/src/aalbanese/datasets/hs/NEON_refl-surf-dir-ortho-mosaic/NEON.D01.HARV.DP3.30006.001.2019-08.basic.20220407T001553Z.RELEASE-2022', '/data/shared/src/aalbanese/datasets/hs/shadow_masks/harv'))
    # import utils
    # IMG = '/data/shared/src/aalbanese/datasets/hs/NEON_refl-surf-dir-ortho-mosaic/NEON.D13.MOAB.DP3.30006.001.2021-04.basic.20220413T132254Z.RELEASE-2022/NEON_D13_MOAB_DP3_645000_4230000_reflectance.h5'
    # rgb = hp.pre_processing(IMG, wavelength_ranges=utils.get_landsat_viz(), merging=True)
    # rgb = hp.make_rgb(rgb["bands"])
    # #rgb = exposure.adjust_gamma(rgb, gamma=0.5)
    # plt.imshow(rgb)
    # plt.show()


    # with ProcessPool(3) as pool:
    #     IN_DIR = ["/data/shared/src/aalbanese/datasets/hs/NEON_refl-surf-dir-ortho-mosaic/NEON.D13.MOAB.DP3.30006.001.2021-04.basic.20220413T132254Z.RELEASE-2022"]
    #     OUT_DIR =  ["/data/shared/src/aalbanese/datasets/hs/crust/moab_crust_2022"]
    #     WAVES = [{'cyano_1': 440,
    #                      'cyano_2': 489,
    #                      'cyano_3': 504,
    #                      'cyano_4': 627,
    #                      'cyano_5': 680,
    #                      'all_crusts': 1450,
    #                      'moss_lichen_1': 1720,
    #                      'generic_crust_1': 1920,
    #                      'moss_lichen_2': 2100,
    #                      'moss_lichen_3': 2180,
    #                      'generic_crust_2': 2300}]
    #     FILES = os.listdir(IN_DIR[0])
    #     IN_DIR *= len(FILES)
    #     OUT_DIR *= len(FILES)
    #     WAVES *= len(FILES)
    #     args = list(zip(FILES,WAVES, IN_DIR, OUT_DIR))
    #     future = pool.map(save_bands, args, timeout=60)
    #     iterator = future.result()
    #     while True:
    #         try:
    #             n = next(iterator)
    #             if isinstance(n, tuple):
    #                 print(n[0])
    #                 np.save(*n)
    #                 #print(n[0])
    #             else:
    #                 print(n)
    #         except TimeoutError as e:
    #             print(e.args)
    #             continue
    #         except StopIteration:
    #             break

    #img_stats('/data/shared/src/aalbanese/datasets/hs/crust/moab_crust_2022', '/data/shared/src/aalbanese/datasets/hs/crust/moab_crust_2022/stats', num_channels=11 )
    # mean = np.load("/data/shared/src/aalbanese/datasets/hs/pca/harv_2022/mean.npy").astype(np.float64)
    # std = np.load("/data/shared/src/aalbanese/datasets/hs/pca/harv_2022/std.npy").astype(np.float64)
    # test = np.load("/data/shared/src/aalbanese/datasets/hs/pca/harv_2022/NEON_D01_HARV_DP3_735000_4713000_reflectance.npy")
    # test = torch.from_numpy(test)
    # norm = tv.transforms.Normalize(mean, std)
    # y = norm(test)
    # print(y)