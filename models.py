from networks import SimSiamResNet
import torch
from torch import nn
import pytorch_lightning as pl
import torchvision.transforms as transforms


class HyperSimSiam(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.network = SimSiamResNet(num_channels=kwargs['num_channels'])
        self.loss = nn.CosineSimilarity(dim=1)

        #TODO: Fix augmentations - Need to adapt to more than 3 channels or PCA images down to 3 channels first
        augmentation = [
        #transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        #transforms.RandomApply([simsiam.loader.GaussianBlur([.1, 2.])], p=0.5),
        transforms.RandomHorizontalFlip()
        #normalize
        ]

        self.augmentation = transforms.Compose(augmentation)

    #TODO: Integrate with KWARGS
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, x, batch_idx):
        #do augmentations then do network
        x = x.squeeze()
        x1, x2  = self.augmentation(x), self.augmentation(x)
        p1, p2, z1, z2 = self.network(x1, x2)
        loss = -(self.loss(p1, z2).mean() + self.loss(p2, z1).mean()) *0.5
        self.log('train_loss', loss)
        return loss

    # TODO: Need species info to do validation
    # def validation_step(self, x, batch_idx):
    #     return None
    
    def forward(self, x):
        return self.network.predict(x)