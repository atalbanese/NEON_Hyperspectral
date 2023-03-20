from sklearn.decomposition import IncrementalPCA
import os
from einops import rearrange
import numpy as np
import numpy.ma as ma
from pebble import ProcessPool
from concurrent.futures import TimeoutError
from tqdm import tqdm
import rasterio as rs
import h5py as hp
import argparse

def bulk_process(pool, fn):
    num_files = len(HS_FILES)
    args_list = list(zip(HS_FILES, CHM_FILES, [PCA_DIR]*num_files, [PCA_SOLVER]*num_files, [HS_DIR]*num_files, [CHM_DIR]*num_files, [SITENAME]*num_files))
    future = pool.map(fn, args_list, timeout=2000)
    iterator = future.result()
    while True:
        try:
            n = next(iterator)
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

def get_hs_filter(bands):
    mask_list = [(bands>=lmin) & (bands<=lmax) for lmin, lmax in HS_FILTERS]
    band_mask = np.logical_or.reduce(mask_list)
    idxs = np.where(band_mask)[0]
    return idxs

def build_inc_pca(hs_file, chm_file):
    hs_file = hp.File(os.path.join(HS_DIR, hs_file), 'r')
    print(hs_file, chm_file)
    bands = hs_file[SITENAME]["Reflectance"]["Metadata"]['Spectral_Data']['Wavelength'][:]
    hs_filter = get_hs_filter(bands)
    img = hs_file[SITENAME]["Reflectance"]["Reflectance_Data"][...,hs_filter]/10000

    chm_open = rs.open(os.path.join(CHM_DIR, chm_file))
    chm = np.squeeze(chm_open.read())
    chm_mask = chm < 2

    bad_mask = np.zeros((1000,1000), dtype=bool)
    for i in range(0, img.shape[-1]):
            z = img[:,:,i]
            y = z>1
            bad_mask += y
    img[bad_mask] = np.nan
    img[chm_mask] = np.nan

    img = rearrange(img, 'h w c -> (h w) c')
    masked = ma.masked_invalid(img)
    to_pca = ma.compress_rows(masked)
    PCA_SOLVER.partial_fit(to_pca)


def do_inc_pca(args):
    hs_file, chm_file, pca_dir, pca_solver, hs_dir, chm_dir, sitename = args

    HS_FILTERS = [[410,1320],[1450,1800],[2050,2485]]
    def get_hs_filter(bands):
        mask_list = [(bands>=lmin) & (bands<=lmax) for lmin, lmax in HS_FILTERS]
        band_mask = np.logical_or.reduce(mask_list)
        idxs = np.where(band_mask)[0]
        return idxs

    pca_file = hs_file.split(".")[0] + '_pca.npy'
    if not os.path.exists(os.path.join(pca_dir, pca_file)):
        hs_file = hp.File(os.path.join(hs_dir, hs_file), 'r')
        bands = hs_file[sitename]["Reflectance"]["Metadata"]['Spectral_Data']['Wavelength'][:]
        hs_filter = get_hs_filter(bands)
        img = hs_file[sitename]["Reflectance"]["Reflectance_Data"][...,hs_filter]/10000

        chm_open = rs.open(os.path.join(chm_dir, chm_file))
        chm = np.squeeze(chm_open.read())
        chm_mask = chm < 2


        bad_mask = np.zeros((1000,1000), dtype=bool)
        for i in range(0, img.shape[-1]):
                z = img[:,:,i]
                y = z>1
                bad_mask += y
        img[bad_mask] = np.nan
        img[chm_mask] = np.nan
        #bare_mask = np.load(os.path.join(mask_dir, bare_file))
        img = rearrange(img, 'h w c -> (h w) c')
        masked = ma.masked_invalid(img)
        mask = masked.mask[:, 0:pca_solver.n_components_]
        to_pca = ma.compress_rows(masked)
        data = pca_solver.transform(to_pca)
        out = np.empty(mask.shape, dtype=np.float64)
        np.place(out, mask, np.nan)
        np.place(out, ~mask, data)
        out = rearrange(out, '(h w) c -> h w c', h=1000, w=1000)
        np.save(os.path.join(pca_dir, pca_file), out)

        return pca_file
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("sitename")
    parser.add_argument("basedir")
    parser.add_argument("-p", "--processes", type=int, default=3)
    args = parser.parse_args()


    SITENAME = args.sitename
    #Windows dir string trick
    BASEDIR = fr"{args.basedir}"

    HS_FILTERS = [[410,1320],[1450,1800],[2050,2485]]
    CHM_DIR = os.path.join(BASEDIR, SITENAME, "CHM")
    HS_DIR = os.path.join(BASEDIR, SITENAME, "HS")
    PCA_DIR = os.path.join(BASEDIR, SITENAME, "PCA")

    CHM_FILES = os.listdir(CHM_DIR)
    HS_FILES = os.listdir(HS_DIR)

    #Sort based on origin so files are paired up
    CHM_FILES.sort(key=lambda x: x.split('_')[-3])
    CHM_FILES.sort(key=lambda x: x.split('_')[-2])

    HS_FILES.sort(key=lambda x: x.split('_')[-3])
    HS_FILES.sort(key=lambda x: x.split('_')[-2])

    PCA_SOLVER = IncrementalPCA(n_components=16)

    print(f'Fitting PCA solver for {SITENAME}')
    for hs_file, chm_file in tqdm(list(zip(HS_FILES, CHM_FILES))):
        build_inc_pca(hs_file, chm_file)
    
    print(f'Generating PCA files for {SITENAME}')
    with ProcessPool(args.processes) as pool:
         bulk_process(pool, do_inc_pca)

    
