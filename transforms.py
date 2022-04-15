import torch
import torchvision.transforms.functional as F
import random

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
