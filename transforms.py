import torch
import torchvision.transforms.functional as F


import random
import numpy as np

#From https://www.sciencedirect.com/science/article/pii/S0034425721000407
class BrightnessAugment(torch.nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
        
    
    def forward(self, arr):
        if torch.rand(1) < self.p:
            change = (1.2 - 0.8) * torch.rand(1).cuda() + 0.8
            a = torch.logit(arr) + torch.logit(change-0.5)
            b = torch.sigmoid(a)
            arr = b
        return arr


class RandomRectangleMask(torch.nn.Module):
    def __init__(self, p=.5):
        super().__init__()
        #_log_api_usage_once(self)
        self.p = p

    def forward(self, arr):
        if torch.rand(1) < self.p:
            zeros = torch.zeros(1)
            upper_bound = arr.shape[-1]
            middle = upper_bound//2
            i = random.randint(0, upper_bound)
            j = random.randint(0, upper_bound)
            width = random.randint(0, upper_bound - i)
            height = random.randint(0, upper_bound - j)

            arr = F.erase(arr, i, j, height, width, zeros)
            if arr[:,:,middle, middle].sum() == 0:
                arr[:,:,middle,middle] = 1
        return arr

class RandomRectangleMaskEven(torch.nn.Module):
    def __init__(self, p=.5):
        super().__init__()
        #_log_api_usage_once(self)
        self.p = p

    def forward(self, arr):
        if torch.rand(1) < self.p:
            zeros = torch.zeros(1)
            upper_bound = arr.shape[-1]
            #middle = upper_bound//2
            i = random.randint(0, upper_bound)
            j = random.randint(0, upper_bound)
            width = random.randint(0, upper_bound - i)
            height = random.randint(0, upper_bound - j)

            arr = F.erase(arr, i, j, height, width, zeros)

        return arr

class RandomPointMaskEven(torch.nn.Module):
    def __init__(self, p=.5):
        super().__init__()
        self.p = p
    
    def forward(self, arr):
        if torch.rand(1) < self.p:
            mask = torch.randint_like(arr, low=0, high=2)
            arr = arr * mask
        return arr

class RandomPointMask(torch.nn.Module):
    def __init__(self, p=.5):
        super().__init__()
        self.p = p
    
    def forward(self, arr):
        if torch.rand(1) < self.p:
            middle = arr.shape[-1]//2
            mask = torch.randint_like(arr, low=0, high=2)
            arr = arr * mask
            if mask[:,:,middle, middle].sum() == 0:
                arr[:,:,middle,middle] = 1
        return arr


class Flip(torch.nn.Module):
    def __init__(self, p=.5):
        super().__init__()
        self.p = p
    
    def forward(self, arr):
        if torch.rand(1) < self.p:
            arr = np.fliplr(arr)
        
        return arr

class BlitDarken(torch.nn.Module):
    def __init__(self, p=.5, d=0.5):
        super().__init__()
        self.p = p
        self.d =d
        self.rng = np.random.default_rng()
    
    def forward(self, arr):
        if torch.rand(1) < self.p:
            middle = arr.shape[-1]//2
            mask = self.rng.integers(2, size=arr.shape)
            mask[:,middle] = 1
            mask = mask.astype(np.float32)
            mask[mask == 0] = self.d
            arr = arr * mask

        return arr


class Blit(torch.nn.Module):
    def __init__(self, p=.5):
        super().__init__()
        self.p = p
        #self.rng = np.random.default_rng()
    
    def forward(self, arr):
        if torch.rand(1) < self.p:
            mask = torch.randint_like(arr, 2)
            arr = arr * mask
        return arr

class Block(torch.nn.Module):
    def __init__(self, p=.5):
        super().__init__()
        self.p = p
        #self.rng = np.random.default_rng()
    
    def forward(self, arr):
        if torch.rand(1) < self.p:
            upper = arr.shape[-1]
            bounds = torch.randint(upper, (2,))
            mask = torch.ones_like(arr)
            mask[bounds.min():bounds.max()] = 0

            arr = arr * mask
        return arr

if __name__ == '__main__':
    flip = Flip(p=1)
    blit = Blit(p=1)
    block = Block(p=1)

    test = np.array([[1, 2, 3, 4, 5], [5, 6, 7, 8, 9]], dtype=np.float32)

    norm = np.array([1, 2, 1, 2, 5])

    print(test)
    print(test/norm)


            
                
