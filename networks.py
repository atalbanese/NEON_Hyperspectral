import torch
import torch.nn as nn
from torchvision import models
import net_gen
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce

# Adapated from https://github.com/facebookresearch/simsiam/blob/main/simsiam/builder.py

class SimSiamUNetFC(nn.Module):
    def __init__(self, num_channels=12, num_classes=10):
        super(SimSiamUNetFC, self).__init__()
        self.encoder = net_gen.ResnetGenerator(num_channels, num_classes)
        self.predictor = Predictor(num_classes)
        # Need to adjust prediction dimensions as well
        

    def forward(self, x1, x2):
        """
        Input:
            x1: first views of images
            x2: second views of images
        Output:
            p1, p2, z1, z2: predictors and targets of the network
            See Sec. 3 of https://arxiv.org/abs/2011.10566 for detailed notations
        """

        # compute features for one view
        z1 = self.encoder(x1) # NxC
        z2 = self.encoder(x2) # NxC

        p1 = self.predictor(z1) # NxC
        p2 = self.predictor(z2) # NxC

        return p1, p2, z1.detach(), z2.detach()

    def predict(self, inp):
        inp = self.encoder(inp)
        return self.predictor(inp)



class Predictor(nn.Module):
    def __init__(self, num_classes):
        super(Predictor, self).__init__()
        self.conv1 = nn.Conv2d(num_classes, num_classes//2, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_classes//2)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(num_classes//2, 1, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(1)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(1, num_classes//2, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(num_classes//2)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(num_classes//2, num_classes, kernel_size=3, padding=1)
        #self.linear1 = nn.Linear(64*64, 64*64, bias=False)
        # self.bn2 = nn.BatchNorm1d(60*60*2)
        # self.relu2 = nn.ReLU()
        # self.linear2 = nn.Linear(60*60*2, 60*60)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.conv4(x)

        return x


#Projector g from https://arxiv.org/pdf/2203.11075.pdf
class DenseProjector(nn.Module):
    def __init__(self, num_channels = 512, num_classes= 10):
        super(DenseProjector, self).__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(num_channels, num_channels, kernel_size=1),
                                    nn.BatchNorm2d(num_channels),
                                    nn.ReLU()
                                    )
        self.layer2 = nn.Sequential(nn.Conv2d(num_channels, num_channels, kernel_size=1),
                                    nn.BatchNorm2d(num_channels),
                                    nn.ReLU()
                                    )
        self.layer3 = nn.Sequential(nn.Conv2d(num_channels, num_channels, kernel_size=1),
                                    nn.BatchNorm2d(num_channels, affine=False),
                                    nn.ReLU()
                                    )
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        return x

#Predictor h - just guessing as to whats this is supposed to be like
class DensePredictor(nn.Module):
    def __init__(self, num_classes =10):
        super(DensePredictor, self).__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(num_classes, num_classes, kernel_size=1),
                                    nn.BatchNorm2d(num_classes),
                                    nn.ReLU()
                                    )
        self.layer2 = nn.Sequential(nn.Conv2d(num_classes, num_classes, kernel_size=1, bias=False),
                                    nn.BatchNorm2d(num_classes),
                                    nn.ReLU()
                                    )
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)

        return x

class DenseProjectorMLP(nn.Module):
    def __init__(self, num_channels=512):
        super(DenseProjectorMLP, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(729, 512, bias=False),
                                    nn.BatchNorm1d(20),
                                    nn.ReLU())
        self.layer2 = nn.Sequential(nn.Linear(512, 512, bias=False),
                                    nn.BatchNorm1d(20),
                                    nn.ReLU())
        self.layer3 = nn.Sequential(nn.Linear(512, 512, bias=False),
                                    nn.BatchNorm1d(20),
                                    nn.ReLU())

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        return x

class DensePredictorMLP(nn.Module):
    def __init__(self):
        super(DensePredictorMLP, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(512, 512, bias=False),
                                    nn.BatchNorm1d(20),
                                    nn.ReLU())
        self.layer2 = nn.Sequential(nn.Linear(512, 512, bias=False),
                                    nn.BatchNorm1d(20, affine=False),
                                    nn.ReLU())


    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)

        return x

class BYOLLinear(nn.Module):
    def __init__(self, start_dims):
        super(BYOLLinear, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(start_dims, 1024),
                                    nn.BatchNorm1d(82),
                                    nn.ReLU(),
                                    nn.Linear(1024, 270))
    
    def forward(self, x):
        return self.layer1(x)

#From https://towardsdatascience.com/implementing-visualttransformer-in-pytorch-184f9f16f632
class PatchEmbedding(nn.Module):
    def __init__(self, in_channels: int = 3, patch_size: int = 16, emb_size: int = 768, img_size: int = 224):
        self.patch_size = patch_size
        super().__init__()
        self.projection = nn.Sequential(
            # using a conv layer instead of a linear one -> performance gains
            nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size),
            Rearrange('b e (h) (w) -> b (h w) e'),
        )
        self.cls_token = nn.Parameter(torch.randn(1,1, emb_size))
        self.positions = nn.Parameter(torch.randn((img_size // patch_size) **2 + 1, emb_size))

        
    def forward(self, x):
        b, _, _, _ = x.shape
        x = self.projection(x)
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)
        # prepend the cls token to the input
        x = torch.cat([cls_tokens, x], dim=1)
        # add position embedding
        #x += self.positions
        return x

class TProjector(nn.Module):
    def __init__(self, num_channels=30, num_classes=60):
        super(TProjector, self).__init__()
        self.layer1 = Rearrange('b (h w) (s1 s2 c) -> b c (h s1) (w s2)', s1=3, s2=3, h=9, w=9)
        self.layer2 = nn.Sequential(nn.Conv2d(num_channels, num_classes, kernel_size=1),
                                    nn.BatchNorm2d(num_classes),
                                    nn.ReLU()
                                    )

        self.layer3 = nn.Sequential(nn.Conv2d(num_classes, num_classes, kernel_size=3, stride=3),
                        nn.BatchNorm2d(num_classes),
                        nn.ReLU())

    def forward(self, x):
        shape = x.shape
        z = x[:,shape[1]-1, :].unsqueeze(1)
        x = x[:,0:shape[1]-1, :]
        x= x+z
        x = self.layer1(x)
        x= self.layer2(x)
        x= self.layer3(x)
        return x

class TPredictor(nn.Module):
    def __init__(self):
        super(TPredictor, self).__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(60, 60, kernel_size=1),
                                    nn.BatchNorm2d(60),
                                    nn.ReLU()
                                    )
        self.layer2 = nn.Sequential(nn.Conv2d(60, 60, kernel_size=1),
                                    nn.BatchNorm2d(60, affine=False))
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x


class SegLinear(nn.Module):
    def __init__(self, num_channels=270, b1=26, b2=26, drop_class=True):
        super(SegLinear, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(num_channels, num_channels),
                                    nn.BatchNorm1d(b1),
                                    nn.ReLU()
                                    )
        self.layer2 = nn.Sequential(nn.Linear(num_channels, num_channels),
                                    nn.BatchNorm1d(b2),
                                    nn.ReLU()
                                    ) 
        self.layer3 = nn.Sequential(nn.Linear(num_channels, num_channels),
                                    nn.BatchNorm1d(b2)
                                    )                                      
        self.drop_class = drop_class



    def forward(self, x):
        if self.drop_class:
            shape = x.shape
            x = x[:,1:shape[1], :]
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        # shape = x.shape
        # # z = x[:,shape[1]-1, :].unsqueeze(1)
        # x = x[:,0:shape[1]-1, :]
        # #x= x*z


        return x

class SegLinearUp(nn.Module):
    def __init__(self, num_in=256, num_out=750, b1=26, b2=26, drop_class=True):
        super(SegLinearUp, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(num_in, 1024),
                                    nn.BatchNorm1d(b1),
                                    nn.ReLU()
                                    )
        self.layer2 = nn.Sequential(nn.Linear(1024, 1024),
                                    nn.BatchNorm1d(b2),
                                    nn.ReLU()
                                    ) 
        self.layer3 = nn.Sequential(nn.Linear(1024, num_out),
                                    nn.BatchNorm1d(b2)
                                    )                                      
        self.drop_class = drop_class



    def forward(self, x):
        if self.drop_class:
            shape = x.shape
            x = x[:,1:shape[1], :]
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        # shape = x.shape
        # # z = x[:,shape[1]-1, :].unsqueeze(1)
        # x = x[:,0:shape[1]-1, :]
        # #x= x*z


        return x

class SegDecoder(nn.Module):
    def __init__(self, num_channels=270, num_classes=60, drop_class=True, patches=81):
        super(SegDecoder, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(num_channels, num_channels//4),
                                    nn.BatchNorm1d(patches),
                                    nn.ReLU()
                                    )
        self.layer2 = nn.Sequential(nn.Linear(num_channels//4, num_channels),
                                    )
        
        
    def forward(self, x):
        
        # z = x[:,shape[1]-1, :].unsqueeze(1)
        
        x = self.layer1(x)
        x = self.layer2(x)

        return x

class DePatch(nn.Module):
    def __init__(self, num_channels=750):
        super(DePatch, self).__init__()
        self.layer0 = nn.Linear(num_channels, 60, bias=False)
        self.layer1 = Rearrange('b (h w) c -> b c h w', h=5, w=5)
        self.layer2 = nn.Upsample(scale_factor=5, mode='bilinear')
        self.layer3 = nn.Softmax(1)

    def forward(self, x):
        x= self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

class SiamLoss(nn.Module):
    def __init__(self):
        super(SiamLoss, self).__init__()
    
    def forward(self, x, y):
        a = x/torch.linalg.norm(x)
        b = y/torch.linalg.norm(y)
        return -(a*b)



class T_Enc(nn.Module):
    def __init__(self, num_channels):
        super(T_Enc, self).__init__()
        encoder_layer = torch.nn.TransformerEncoderLayer(d_model=25*num_channels, nhead=15, dim_feedforward=1024, batch_first=True)
        self.layer_1 = nn.Sequential(PatchEmbedding(in_channels=num_channels, patch_size=5, emb_size=25*num_channels, img_size=25),
                                    torch.nn.TransformerEncoder(encoder_layer, num_layers=4),
                                    SegLinear(num_channels=25 * num_channels, b1=25, b2=25))
        self.layer_2 = SegDecoder(num_channels = 25 * num_channels, patches=25)

    def forward(self, x):
        z_1 = self.layer_1(x)
        p_1 = self.layer_2(z_1)

        return p_1, z_1.detach()


class C_Enc(nn.Module):
    def __init__(self, num_channels):
        super(C_Enc, self).__init__()
        self.layer_1 = nn.Sequential(net_gen.ResnetEncoder(num_channels, num_channels),
                                    Rearrange('b c h w -> b (h w) c'),
                                    SegLinearUp(b1=25, b2=25, drop_class=False))
        self.layer_2 = SegDecoder(num_channels = 25 * num_channels, patches=25)

    def forward(self, x):
        z_1 = self.layer_1(x)
        p_1 = self.layer_2(z_1)

        return p_1, z_1.detach()





if __name__ == "__main__":
    s = SegLinear()
    p = torch.ones(256,82,270)
    q = s(p)
    print(q.shape)



