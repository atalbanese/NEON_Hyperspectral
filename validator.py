import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import h5_helper as hp

class Validator():
    def __init__(self, **kwargs):
        self.file = kwargs["file"]
        self.imgs_dir = kwargs["img_dir"]
        self.valid_data = self.open_file()
        self.num_clusters = kwargs["num_clusters"]
        self.cluster_dict = self.make_empty_dict()

    def open_file(self):
        data = pd.read_csv(self.file, usecols=['easting', 'northing', 'taxonID', 'ninetyCrownDiameter'])
        data = data.dropna()
        data["pixels"] = (data['ninetyCrownDiameter']**2)//1
        data["file_west_bound"] = data["easting"] - data["easting"] % 1000
        data["file_south_bound"] = data["northing"] - data["northing"] % 1000
        data['x_min'] = (data["easting"] %1000-data["ninetyCrownDiameter"]/2)//1
        data['y_min'] = ((1000-data["northing"] %1000)-data["ninetyCrownDiameter"]/2)//1
        data['x_max'] = data['x_min'] + data["ninetyCrownDiameter"]//1
        data['y_max'] = data['y_min'] + data["ninetyCrownDiameter"]//1
        index_names = data[(data['x_min'] <0) | (data['y_min']<0) | (data['x_max'] >999) | (data['y_max']>999)].index
        data = data.drop(index_names)
        data = data.astype({"file_west_bound": int,
                            "file_south_bound": int,
                            'x_min': int,
                            'x_max': int,
                            'y_min': int,
                            'y_max': int})

        data = data.astype({"file_west_bound": str,
                            "file_south_bound": str})

        data['file_coords'] = data['file_west_bound'] + ',' + data['file_south_bound']

        return data

    def make_empty_dict(self):
        keys = self.valid_data["taxonID"].unique()
        template = {key:{i:0 for i in range(0,self.num_clusters)} for key in keys}
        return template


    def validate(self, file_coords, file_loc):
        valid = self.valid_data.loc[self.valid_data["file_coords"] == file_coords]
        if len(valid.index>0):
            clustered = np.load(file_loc)
            for ix, row in valid.iterrows():
                select = clustered[row['x_min']:row['x_max'], row['y_min']:row['y_max']]
                taxon = row['taxonID']
                groups, counts = np.unique(select, return_counts=True)
                for j, group in np.ndenumerate(groups):
                    self.cluster_dict[taxon][group] += counts[j]




def pca(data):
    data = hp.get_features(data)
    #data = stack_all(data)
    print("doing pca")
    data = hp.pca(data, n_components=8, whiten=False)
    return data




if __name__ == "__main__":
    NUM_CLUSTERS = 4
    h5_file = "/data/shared/src/aalbanese/datasets/hs/NEON_refl-surf-dir-ortho-mosaic/NEON.D16.WREF.DP3.30006.001.2021-07.basic.20220330T192306Z.PROVISIONAL/NEON_D16_WREF_DP3_580000_5075000_reflectance.h5"
    coords = "580000,5075000"
    valid = Validator(file="/data/shared/src/aalbanese/datasets/neon_wref.csv", img_dir="", num_clusters=NUM_CLUSTERS)

    opened = hp.pre_processing(h5_file, get_all=True)["bands"]
    reduce_dim = pca(opened)
    clustered = hp.ward_cluster(reduce_dim, NUM_CLUSTERS, n_neighbors=8)
    clustered = clustered.reshape((1000,1000))

    np.save("test_cluster_580000_5075000_4.npy", clustered)

    
    plt.imshow(np.load("test_cluster_580000_5075000_4.npy"))
    plt.show()
    valid.validate(coords, "test_cluster_580000_5075000_4.npy")
    print(valid.cluster_dict)
    print("here")
