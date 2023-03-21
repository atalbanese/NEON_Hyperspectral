import torch
import torchvision.transforms.functional as F
import random
import numpy as np

class NormalizeHS(torch.nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.mean = mean
        self.std = std
    
    def forward(self, arr):
        return (arr-self.mean)/self.std

#From https://www.sciencedirect.com/science/article/pii/S0034425721000407
class BrightnessAugment(torch.nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
        
    
    def forward(self, arr):
        if torch.rand(1) < self.p:
            change = (1.2 - 0.8) * torch.rand(1) + 0.8
            a = torch.logit(arr) + torch.logit(change-0.5)
            b = torch.sigmoid(a)
            arr = b
        return arr


class Flip(torch.nn.Module):
    def __init__(self, p=.5):
        super().__init__()
        self.p = p
    
    def forward(self, arr):
        if torch.rand(1) < self.p:
            arr = np.fliplr(arr)
        
        return arr

class Blit(torch.nn.Module):
    def __init__(self, p=.5):
        super().__init__()
        self.p = p
        self.missing = 0
        #self.rng = np.random.default_rng()
    
    def forward(self, arr):
        if torch.rand(1) < self.p:
            mask = torch.randint_like(arr, 2)
            arr[mask == 0] = self.missing
        return arr


class Block(torch.nn.Module):
    def __init__(self, p=.5):
        super().__init__()
        self.p = p
        self.missing = 0
        #self.rng = np.random.default_rng()
    
    def forward(self, arr):
        if torch.rand(1) < self.p:
            upper = arr.shape[-1]
            bounds = torch.randint(upper, (2,))
            mask = torch.ones_like(arr)
            mask[bounds.min():bounds.max()] = 0

            arr[mask == 0] = self.missing
        return arr




            
                
