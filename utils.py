import h5_helper as hp
import numpy as np

from sklearn.decomposition import PCA
from sklearn.neighbors import kneighbors_graph
from sklearn import cluster

import torch
import torchvision as tv

def get_features(inp, feature_band=2):
    return np.reshape(inp, (-1,inp.shape[feature_band]))


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