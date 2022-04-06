import torch
import torchvision.transforms.functional as F

class RandomRectangleMask(torch.nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        #_log_api_usage_once(self)
        self.p = p

    def forward(self, arr):
        if torch.rand(1) < self.p:
            zeros = torch.zeros_like(arr)
            upper_bound = arr.shape[-1]
            middle = upper_bound//2
            i = torch.rand(1) * upper_bound
            j = torch.rand(1) * upper_bound
            width = torch.rand(1) * (upper_bound - i)
            height = torch.rand(1) * (upper_bound -j)

            arr = F.erase(arr, i, j, height, width, zeros)
            if arr[:,middle, middle] == 0:
                arr[:,middle,middle] = 1
        return arr


class RandomPointMask(torch.nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
    
    def forward(self, arr):
        if torch.rand(1) < self.p:
            middle = arr.shape[-1]//2
            mask = torch.randint_like(arr, low=0, high=2)
            arr = arr * mask
            if mask[:,middle, middle] == 0:
                arr[:,middle,middle] = 1
        return arr
