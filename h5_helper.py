from select import select
import h5py as hp
import numpy as np
import matplotlib.pyplot as plt
from skimage import exposure
import matplotlib.patches as mpatches
from sklearn.decomposition import PCA, IncrementalPCA


def pca(band_arr, **kwargs):
    pca = PCA(**kwargs)
    pca.fit(band_arr)
    return pca.transform(band_arr)

def get_features(inp, feature_band=2):
    return np.reshape(inp, (-1,inp.shape[feature_band]))


#Finds index of nearest wavelength
def find_nearest(dataset, search_val):
    diff_arr = np.absolute(dataset[:]-search_val)
    return diff_arr.argmin()


#Gets band data, fixes nan values and normalizes to 0-1
def get_band(dataset, band_index):
    extracted = dataset[:,:,band_index]
    extracted = extracted.astype(np.float32)
    extracted[extracted == -9999] = np.nan
    return extracted / 10000

# Given an upper and lower value, make a mask from the spectral bands metadata which captures those values
def get_filter_range(dataset, lower, upper):
    return np.logical_and(dataset[:] >= lower, dataset[:] <= upper)

# Filter hyperspectral dataset by boolean mask for specific wavelengths, merge, and normalize to 0-1
def filter_and_merge(dataset, mask, merging=False):
    #Mask nonzero works around a bug in h5py where you cant filter using a boolean mask
    extracted = dataset[:,:,mask.nonzero()[0]]
    extracted = extracted.astype(np.float32)
    extracted[extracted == -9999] = np.nan
    #Is mean the right way to merge hyperspectral wavelengths?

    merged = np.mean(extracted, axis=2)
    # Normalize to 0 to 1 range, all these neon images are 0 to 10000
    merged /= 10000
    return merged


#Take a dictionary of bands and return a numpy array thats [:,:,3] in red, green, blue order
def make_rgb(band_dict):
    to_stack = [band_dict["red"],
                band_dict["green"],
                band_dict["blue"]]
    return np.stack(to_stack, axis=2)

def stack_all(band_dict, axis=2):
    to_stack = [value for value in band_dict.values()]
    return np.stack(to_stack, axis=axis)

#TODO: this is getting unwieldy. Split it up
#Open hyperspectral file, get the supplied wavelengths, merge them into bands, and return them along with metadata for the file
def pre_processing(f, wavelength_ranges=None, mosaic=True, merging=False, select_bands=None, get_all=False):
    was_str = False
    to_return = {}
    if isinstance(f, str):
        was_str = True
        f = hp.File(f, 'r')
    #Get sitename
    group_key = list(f.keys())[0]

    #Get values out of h5
    meta_data = f[group_key]["Reflectance"]["Metadata"]
    refl_values = f[group_key]["Reflectance"]["Reflectance_Data"]

    if mosaic:
        zenith = meta_data['to-sensor_zenith_angle']
        spectral = meta_data['Spectral_Data']
        #TODO: FIx Missing angles (-1)
        # angles = get_solar_stats(meta_data)
        # angles['sensor_zenith'] = meta_data['to-sensor_zenith_angle'][:]
        # angles['azimuth'] = np.abs(meta_data['to-sensor_azimuth_angle'][:]-180.0-angles['azimuth'])
    else:
        angles = 'Getting angle metadata from flightline files still in development'
    
    angles = 'Getting angles still in dev'

    #angles['azimuth'] = np.ones_like(angles['sensor_zenith']) *10
    #plt.imshow(zenith)
    #plt.show()
    spectral_bands = meta_data['Spectral_Data']['Wavelength']
    meta_data = {"map_info": meta_data['Coordinate_System']['Map_Info'][()].decode("utf-8"),
                    "proj": meta_data['Coordinate_System']['Proj4'][()].decode("utf-8"),
                    "epsg": meta_data['Coordinate_System']['EPSG Code'][()].decode("utf-8")}
    to_return["meta"] = meta_data
    if not get_all:
        if merging:
            #Get masks to select specific bands as defined by wavelength_ranges
            band_ranges = {band: get_filter_range(spectral_bands, band_range["lower"], band_range["upper"]) 
                            for band, band_range in wavelength_ranges.items()}

            #Filter and merge hyperspectral bands into blue, gree, red, nir bands
            band_data = {band: filter_and_merge(refl_values, band_range) for band, band_range in band_ranges.items()}
        else:
            if select_bands is None:
                select_bands = {band: find_nearest(spectral_bands, wavelength) for band, wavelength in wavelength_ranges.items()}
            band_data = {band: get_band(refl_values, band_index) for band, band_index in select_bands.items()}
        #to_return = [band_data, meta_data, angles]
        to_return["bands"] = band_data
        
        if select_bands is not None:
            to_return["selected"] =select_bands

    else:
        refl_values = refl_values[:]/10000
        refl_values[refl_values<0] = -1
        to_return["bands"] = refl_values
    if was_str:
        f.close()
    return to_return


def get_solar_stats(metadata):
    # Get index mask - each integer represents a specific flight log, indexed by Data_Files
    data_index = metadata["Ancillary_Imagery"]["Data_Selection_Index"]
    # Get flight logs index
    data_index_list = data_index.attrs['Data_Files'].split(',')
    data_index_list = [index.split('_')[-2] for index in data_index_list]

    # Get actual flight logs and extract solar angles
    flight_logs = metadata["Logs"].keys()

    flight_angles = {key:{'azimuth': metadata["Logs"][key]['Solar_Azimuth_Angle'][()],
                          'zenith': metadata["Logs"][key]['Solar_Zenith_Angle'][()],
                          'flight_time': key} for key in flight_logs}

    # Correlate flight logs index to solar angles from flight logs
    flight_index = {i:flight_angles[data_index_list[i]] for i in range(len(data_index_list)) if data_index_list[i] in flight_angles}

    #Get actual index integer values
    index_values = data_index[:]
    #Get unique indexes and index inversion map
    u,inv = np.unique(index_values, return_inverse = True)

    #Use unique indexes to get angles from flights and then remap to original using index inversion map
    zenith_map = np.array([flight_index[x]["zenith"] for x in u])[inv].reshape(index_values.shape)
    azimuth_map = np.array([flight_index[x]["azimuth"] for x in u])[inv].reshape(index_values.shape)

    return {'zenith':zenith_map, 'azimuth':azimuth_map}


def get_all_data(file_location):
    dataset_list = []

    def extract_dataset(h_group):
        if isinstance(h_group, hp._hl.group.Group):
            for key in list(h_group.keys()):
                extract_dataset(h_group[key])
        else:
            dataset_list.append(h_group)

    
    with hp.File(file_location, 'r') as f:
        group_key = list(f.keys())[0]
        all_values = f[group_key]
        extract_dataset(all_values)
    
        return dataset_list


def skip_downsample(inp, step_size):
    return inp[::step_size, ::step_size]

def down_all(inp, step_size):
    return {key: skip_downsample(value, step_size) for key, value in inp.items()}

if __name__ == "__main__":
    select_bands = {"i":i for i in range(0,426)}
    h5_file = "/data/shared/src/aalbanese/datasets/hs/NEON_refl-surf-dir-ortho-mosaic/NEON.D16.WREF.DP3.30006.001.2021-07.basic.20220330T192306Z.PROVISIONAL/NEON_D16_WREF_DP3_590000_5077000_reflectance.h5"

    data,_,_ = pre_processing(h5_file, get_all=True)
    data = get_features(data)
    #data = stack_all(data)
    lower_dim, var = pca(data, n_components=30, whiten=True)
    print(lower_dim)


