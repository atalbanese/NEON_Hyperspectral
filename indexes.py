import numpy as np
import h5_helper as hp
import os
class IndexHolder():
    def __init__(self, in_dir, f):
        self.bands = {860, 650, 470, 531, 570, 1754, 1680, 1510, 1680, 850, 550, 1660, 714, 752, 733, 2210, 2090, 2280, 700, 710, 544,
                        677, 750, 542,800,670,415,435,531,570,880,673,695,554,445,740,720}
        self.bands = {f'{v}':v for v in self.bands}
        self.inp =  hp.pre_processing(os.path.join(in_dir, f), wavelength_ranges=self.bands())["bands"]
    
    def ndvi(self):
        return (self.inp['860'] - self.inp['650'])/(self.inp['860'] + self.inp['650'])
    
    def evi(self):
        return (2.5 * (self.inp['860']-self.inp['650']))/(self.inp['860'] + (6 * self.inp['650']) - (7.5 * self.inp['470'])+1)
    
    def arvi(self, gamma):
        return (self.inp['860'] - (self.inp['650'] - (gamma*(self.inp['470'] - self.inp['650']))))/(self.inp['860'] + (self.inp['650'] - (gamma*(self.inp['470'] - self.inp['650']))))

    def pri(self):
        return (self.inp['531'] - self.inp['570'])/(self.inp['531'] + self.inp['570'])

    def ndli(self):
        return (np.log10(1/self.inp['1754']) - np.log10(1/self.inp['1680']))/(np.log10(1/self.inp['1754']) + np.log10(1/self.inp['1680']))

    def ndni(self):
        return (np.log10(1/self.inp['1510']) - np.log10(1/self.inp['1680']))/(np.log10(1/self.inp['1510']) + np.log10(1/self.inp['1680']))

    def savi(self, L):
        return ((1+L)*(self.inp['850'] - self.inp['650']))/(self.inp['850'] + self.inp['650'] + L)

    def aci_2(self):
        return self.inp['650']/self.inp['550']

    def dwsi_2(self):
        return self.inp['1660']/self.inp['550']

    def rvsi(self):
        return (self.inp['714']+self.inp['752'])/2 - self.inp['733']
    
    def swir_vi(self):
        return 37.73*(self.inp['2210'] - self.inp['2090']) + 26.27*(self.inp['2280'] - self.inp['2090']) + 0.57
    
    def ari_1(self):
        return (1/self.inp['550']) - (1/self.inp['700'])
    
    def ari_2(self):
        return self.inp['800'] * self.ari_1()
    
    def cri_2(self):
        return (1/self.inp['510']) - (1/self.inp['550'])

    def datt_1(self):
        return (self.inp['850'] - self.inp['710'])/(self.inp['850'] - self.inp['680'])
    
    #Need to determine what datt_2 is

    def gi(self):
        return self.inp['554']/self.inp['677']

    #need to determine what dvi is

    def gm_1(self):
        return self.inp['750']/self.inp['550']
    
    def gm_2(self):
        return self.inp['750']/self.inp['700']

    def mac(self):
        return self.inp['542']/self.inp['750']

    def msr(self):
        return ((self.inp['800']/self.inp['670'])-1)/np.sqrt((self.inp['800']/self.inp['670'])+1)

    #mtci might not apply at high res

    def mtvi(self):
        return 1.2 * ((1.2 * (self.inp['800'] - self.inp['550'])) - 2.5*(self.inp['670'] - self.inp['550']))

    def mtvi_2(self):
        top =  1.5 * ((1.2 * (self.inp['800'] - self.inp['550'])) - 2.5*(self.inp['670'] - self.inp['550']))
        bottom = np.sqrt((2*self.inp['800'] +1)**2.0 - (6*self.inp['800'] - 5*(self.inp['670']**0.5))-0.5)
        return

    #have to calculate derivitave, figure it out later
    #def mrendvi(self):

    def npqi(self):
        return (self.inp['415']-self.inp['435']) / (self.inp['415']+self.inp['435'])     
    
    def pri(self):
        return (self.inp['531']-self.inp['570']) / (self.inp['531']+self.inp['570'])  
    
    def rdvi(self):
        return np.sqrt(((self.inp['880'] - self.inp['673'])**2.0)/(self.inp['880'] + self.inp['673']))


    #need to look into reip - calculate over training data

    def rgri(self):
        return self.inp['695']/self.inp['554']

    def sipi(self):
        return (self.inp['800'] - self.inp['445'])/(self.inp['800'] + self.inp['445'])

    def vog(self):
        return self.inp['740']/self.inp['720']    





    


    


    