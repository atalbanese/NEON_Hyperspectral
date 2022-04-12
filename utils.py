import h5_helper as hp
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.neighbors import kneighbors_graph
from sklearn import cluster
import os
from tqdm import tqdm
import torch
import torchvision as tv
from multiprocessing import Pool

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

def plot_output_files(dir):
    for file in os.listdir(dir):
        if ".npy" in file:
            y = np.load(os.path.join(dir,file))
            plt.imshow(y)
            plt.show()

def bulk_pca(file, in_dir, out_dir):
    band = {'b1': 500}
 
    if ".h5" in file:
        new_file = file.split(".")[0] + ".npy"
        if not os.path.exists(os.path.join(out_dir,new_file)):
            img = hp.pre_processing(os.path.join(in_dir,file), wavelength_ranges=band)
            if np.isnan(img["bands"]['b1']).sum() == 0:
                img = hp.pre_processing(os.path.join(in_dir,file), get_all=True)["bands"]
                img = pca(img, True, n_components=30)
                np.save(os.path.join(out_dir,new_file), img)

def img_stats(dir):
    files = os.listdir(dir)
    psum = np.zeros((30,), dtype=np.float128) 
    sq = np.zeros((30,), dtype=np.float128) 
    num_files = 0
    for file in tqdm(files):
        if ".npy" in file:
            num_files += 1
            img = np.load(os.path.join(dir, file))
            img = np.reshape(img, (30, -1))
            sum = img.sum(axis=1)
            psum += sum
            sq += (img**2).sum(axis=1)
    count = num_files *1000 * 1000
    total_mean = psum/count
    total_var = (sq/count) - (total_mean**2)
    total_std = np.sqrt(total_var)

    np.save(os.path.join(dir, "mean.npy"), total_mean)
    np.save(os.path.join(dir, "std.npy"), total_std)




if __name__ == '__main__':
    #bulk_pca(
    # with Pool(6) as pool:
    #     in_dir = ["/data/shared/src/aalbanese/datasets/hs/NEON_refl-surf-dir-ortho-mosaic/NEON.D01.HARV.DP3.30006.001.2019-08.basic.20220407T001553Z.RELEASE-2022"]
    #     out_dir =  ["/data/shared/src/aalbanese/datasets/hs/pca/harv_2022"]
    #     files = os.listdir(in_dir[0])
    #     in_dir *= len(files)
    #     out_dir *= len(files)
    #     args = list(zip(files, in_dir, out_dir))
    #     pool.starmap(bulk_pca, args)

    #img_stats("/data/shared/src/aalbanese/datasets/hs/pca/harv_2022")
    mean = np.load("/data/shared/src/aalbanese/datasets/hs/pca/harv_2022/mean.npy").astype(np.float64)
    std = np.load("/data/shared/src/aalbanese/datasets/hs/pca/harv_2022/std.npy").astype(np.float64)
    test = np.load("/data/shared/src/aalbanese/datasets/hs/pca/harv_2022/NEON_D01_HARV_DP3_735000_4713000_reflectance.npy")
    test = torch.from_numpy(test)
    norm = tv.transforms.Normalize(mean, std)
    y = norm(test)
    print(y)