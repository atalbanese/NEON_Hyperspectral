import torch


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

class Blit(torch.nn.Module):
    def __init__(self, missing, p=.5):
        super().__init__()
        self.p = p
        self.missing = missing
        #self.rng = np.random.default_rng()
    
    def forward(self, arr):
        if torch.rand(1) < self.p:
            mask = torch.randint_like(arr, 2)
            arr[mask == 0] = self.missing
        return arr

class PatchBlock(torch.nn.Module):
    def __init__(self,missing, p=0.5):
        super().__init__()
        self.p = p
        self.missing = missing
        #self.rng = np.random.default_rng()
    
    def forward(self, arr):
        if torch.rand(1) < self.p:
            b, s, _ = arr.shape
            mask = torch.rand((b, s), dtype=torch.float32, device=arr.device)
            mask = mask >= 0.4
            #mask = mask.unsqueeze(2)

           # arr = arr * mask
            arr[~mask] = self.missing
        return arr



class Block(torch.nn.Module):
    def __init__(self, missing, p=.5):
        super().__init__()
        self.p = p
        self.missing = missing
        #self.rng = np.random.default_rng()
    
    def forward(self, arr):
        if torch.rand(1) < self.p:
            upper = arr.shape[-1]
            bounds = torch.randint(upper, (2,))
            mask = torch.ones_like(arr)
            mask[bounds.min():bounds.max()] = 0

            arr[mask == 0] = self.missing
        return arr
