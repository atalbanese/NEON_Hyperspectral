import torch
import torch.nn as nn
from torchvision import models
import net_gen

# Adapated from https://github.com/facebookresearch/simsiam/blob/main/simsiam/builder.py

class SimSiamUNetFC(nn.Module):
    def __init__(self, num_channels=12, num_classes=10):
        super(SimSiamUNetFC, self).__init__()
        self.encoder = net_gen.UnetGenerator(num_channels, num_classes, 3)
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
        self.layer1 = nn.Sequential(nn.Linear(4096, 512, bias=False),
                                    nn.BatchNorm1d(10),
                                    nn.ReLU())
        self.layer2 = nn.Sequential(nn.Linear(512, 512, bias=False),
                                    nn.BatchNorm1d(10),
                                    nn.ReLU())
        self.layer3 = nn.Sequential(nn.Linear(512, 512, bias=False),
                                    nn.BatchNorm1d(10),
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
                                    nn.BatchNorm1d(10),
                                    nn.ReLU())
        self.layer2 = nn.Sequential(nn.Linear(512, 512, bias=False),
                                    nn.BatchNorm1d(10, affine=False),
                                    nn.ReLU())


    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)

        return x

 



