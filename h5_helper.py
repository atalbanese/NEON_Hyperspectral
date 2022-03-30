import h5py as hp
import numpy as np
import matplotlib.pyplot as plt
from skimage import exposure
import matplotlib.patches as mpatches

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

#Open hyperspectral file, get the supplied wavelengths, merge them into bands, and return them along with metadata for the file
def pre_processing(file_location, wavelength_ranges, mosaic=True, merging=True):
    with hp.File(file_location, 'r') as f:
        #Get sitename
        group_key = list(f.keys())[0]

        #Get values out of h5
        meta_data = f[group_key]["Reflectance"]["Metadata"]
        refl_values = f[group_key]["Reflectance"]["Reflectance_Data"]
        if mosaic:
            zenith = meta_data['to-sensor_zenith_angle']
            spectral = meta_data['Spectral_Data']
            angles = get_solar_stats(meta_data)
            angles['sensor_zenith'] = meta_data['to-sensor_zenith_angle'][:]
            angles['azimuth'] = np.abs(meta_data['to-sensor_azimuth_angle'][:]-180.0-angles['azimuth'])
        else:
            angles = 'Getting angle metadata from flightline files still in development'

        #angles['azimuth'] = np.ones_like(angles['sensor_zenith']) *10
        #plt.imshow(zenith)
        #plt.show()
        spectral_bands = meta_data['Spectral_Data']['Wavelength']
        if merging:
            #Get masks to select specific bands as defined by wavelength_ranges
            band_ranges = {band: get_filter_range(spectral_bands, band_range["lower"], band_range["upper"]) 
                            for band, band_range in wavelength_ranges.items()}

            #Filter and merge hyperspectral bands into blue, gree, red, nir bands
            band_data = {band: filter_and_merge(refl_values, band_range) for band, band_range in band_ranges.items()}
        else:
            select_bands = {band: find_nearest(spectral_bands, wavelength) for band, wavelength in wavelength_ranges.items()}
            band_data = {band: get_band(refl_values, band_index) for band, band_index in select_bands.items()}

    return band_data, meta_data, angles


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


