import numpy as np
import h5_helper as hp
import os



BANDS = {860, 650, 470, 531, 570, 1754, 1680, 1510, 1680, 850, 550, 1660, 714, 752, 733, 2210, 2090, 2280, 700, 710, 544,
                677, 750, 542,800,670,415,435,531,570,880,673,695,554,445,740,720, 510, 680}
BANDS = {f'{v}':v for v in BANDS}



#inp =  hp.pre_processing(os.path.join(in_dir, f), wavelength_ranges=bands())["bands"]

def ndvi(inp):
    return (inp['860'] - inp['650'])/(inp['860'] + inp['650'])

def evi(inp):
    return (2.5 * (inp['860']-inp['650']))/(inp['860'] + (6 * inp['650']) - (7.5 * inp['470'])+1)

# #Need default values for these
# def arvi(inp, gamma):
#     return (inp['860'] - (inp['650'] - (gamma*(inp['470'] - inp['650']))))/(inp['860'] + (inp['650'] - (gamma*(inp['470'] - inp['650']))))

def pri(inp):
    return (inp['531'] - inp['570'])/(inp['531'] + inp['570'])

def ndli(inp):
    return (np.log10(1/inp['1754']) - np.log10(1/inp['1680']))/(np.log10(1/inp['1754']) + np.log10(1/inp['1680']))

def ndni(inp):
    return (np.log10(1/inp['1510']) - np.log10(1/inp['1680']))/(np.log10(1/inp['1510']) + np.log10(1/inp['1680']))

# #Need default values
# def savi(inp, L):
#     return ((1+L)*(inp['850'] - inp['650']))/(inp['850'] + inp['650'] + L)

def aci_2(inp):
    return inp['650']/inp['550']

def dwsi_2(inp):
    return inp['1660']/inp['550']

def rvsi(inp):
    return (inp['714']+inp['752'])/2 - inp['733']

def swir_vi(inp):
    return 37.73*(inp['2210'] - inp['2090']) + 26.27*(inp['2280'] - inp['2090']) + 0.57

def ari_1(inp):
    return (1/inp['550']) - (1/inp['700'])

def ari_2(inp):
    return inp['800'] * ari_1(inp)

def cri_2(inp):
    return (1/inp['510']) - (1/inp['550'])

def datt_1(inp):
    return (inp['850'] - inp['710'])/(inp['850'] - inp['680'])

#Need to determine what datt_2 is

def gi(inp):
    return inp['554']/inp['677']

#need to determine what dvi is

def gm_1(inp):
    return inp['750']/inp['550']

def gm_2(inp):
    return inp['750']/inp['700']

def mac(inp):
    return inp['542']/inp['750']

def msr(inp):
    return ((inp['800']/inp['670'])-1)/np.sqrt((inp['800']/inp['670'])+1)

#mtci might not apply at high res

def mtvi(inp):
    return 1.2 * ((1.2 * (inp['800'] - inp['550'])) - 2.5*(inp['670'] - inp['550']))

def mtvi_2(inp):
    top =  1.5 * ((1.2 * (inp['800'] - inp['550'])) - 2.5*(inp['670'] - inp['550']))
    bottom = np.sqrt((2*inp['800'] +1)**2.0 - (6*inp['800'] - 5*(inp['670']**0.5))-0.5)
    return top/bottom

#have to calculate derivitave, figure it out later
#def mrendvi(inp):

def npqi(inp):
    return (inp['415']-inp['435']) / (inp['415']+inp['435'])     

def pri(inp):
    return (inp['531']-inp['570']) / (inp['531']+inp['570'])  

def rdvi(inp):
    return np.sqrt(((inp['880'] - inp['673'])**2.0)/(inp['880'] + inp['673']))


#need to look into reip - calculate over training data

def rgri(inp):
    return inp['695']/inp['554']

def sipi(inp):
    return (inp['800'] - inp['445'])/(inp['800'] + inp['445'])

def vog(inp):
    return inp['740']/inp['720']    

INDEX_FNS = [ndvi, evi, pri, ndli, ndni, aci_2, dwsi_2, rvsi, swir_vi, ari_1, ari_2, cri_2, datt_1,
                 gi, gm_1, gm_2, mac, msr, mtvi, mtvi_2, npqi, pri, rdvi, rgri, sipi, vog]




    


    


    