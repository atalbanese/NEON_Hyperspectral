import utils
from einops import rearrange
import os
import random
import numpy as np
import h5_helper as hp

def cluster_img(img):
    img = np.load(img, allow_pickle=True)
    img[img != img] = -1
    #img = hp.stack_all(img)
    features = rearrange(img, 'h w c -> (h w) c')
    cluster = utils.ward_cluster(features, 5, n_neighbors=6)
    cluster = rearrange(cluster, '(h w) -> h w', h=1000, w=1000)
    return cluster

def cluster_imgs(img_folder, out_folder, num_imgs):
    files = [file for file in os.listdir(img_folder) if ".npy" in file]
    sample = random.sample(files, num_imgs)
    for file in sample:
        out = cluster_img(os.path.join(img_folder, file))
        np.save(os.path.join(out_folder, file), out)




if __name__ == "__main__":
    CRUST_IMGS = '/data/shared/src/aalbanese/datasets/hs/crust/moab_crust_2022'
    CLUSTER_DIR = '/data/shared/src/aalbanese/datasets/hs/crust/ward_clustered'
    cluster_imgs(CRUST_IMGS, CLUSTER_DIR, 10)

