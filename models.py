from random import sample
import networks
import torch
from torch import nn, norm
import pytorch_lightning as pl
import torchvision as tv



# class HyperSimSiam(pl.LightningModule):
#     def __init__(self, **kwargs):
#         super().__init__()
#         self.kwargs = kwargs
#         self.network = SimSiamResNet(num_channels=kwargs['num_channels'])
#         self.loss = nn.CosineSimilarity(dim=1)

#         #TODO: Fix augmentations - Need to adapt to more than 3 channels or PCA images down to 3 channels first
#         augmentation = [
#         #transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
#         transforms.RandomApply([
#             transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
#         ], p=0.8),
#         transforms.RandomGrayscale(p=0.2),
#         #transforms.RandomApply([simsiam.loader.GaussianBlur([.1, 2.])], p=0.5),
#         transforms.RandomHorizontalFlip()
#         #normalize
#         ]

#         self.augmentation = transforms.Compose(augmentation)

#     #TODO: Integrate with KWARGS
#     def configure_optimizers(self):
#         optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
#         return optimizer

#     def training_step(self, x, batch_idx):
#         #do augmentations then do network
#         x = x.squeeze()
#         x1, x2  = self.augmentation(x), self.augmentation(x)
#         p1, p2, z1, z2 = self.network(x1, x2)
#         loss = -(self.loss(p1, z2).mean() + self.loss(p2, z1).mean()) *0.5
#         self.log('train_loss', loss)
#         return loss

#     # TODO: Need species info to do validation
#     # def validation_step(self, x, batch_idx):
#     #     return None
    
#     def forward(self, x):
#         return self.network.predict(x)

class HyperSimSiamWaveAugment(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.network = networks.SimSiamUNetFC(num_channels=kwargs['num_channels'])
        #self.loss = nn.CosineSimilarity(dim=1)
        self.loss = nn.CrossEntropyLoss()
        #self.loss=nn.BCEWithLogitsLoss()

    #TODO: Integrate LR with KWARGS
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=5e-4)
        return optimizer

    def training_step(self, x, batch_idx):
        

        viz, inp, inp_aug = x["viz"].squeeze(), x["base"].squeeze(), x["rand"].squeeze()
        
        sample_imgs = viz[:6]
        self.save_grid(sample_imgs, 'rgb')
        
        p1, p2, z1, z2 = self.network(inp, inp_aug)    

        sample_pred = p1[:6]
        sample_pred = self.get_classifications(sample_pred)
        self.save_grid(sample_pred, 'predicted')

        sample_proj = z1[:6]
        sample_proj = self.get_classifications(sample_proj)
        self.save_grid(sample_proj, 'projected')

        loss = (self.loss(p1, z2).mean() + self.loss(p2, z1).mean()) *0.5
        self.log('train_loss', loss)
        return loss

    def save_grid(self, inp, name):
        tb = self.logger.experiment
        img_grid = tv.utils.make_grid(inp, normalize=True, scale_each=True)
        tb.add_image(name, img_grid, self.current_epoch)



    # TODO: Need species info to do validation
    # def validation_step(self, x, batch_idx):
    #     return None

    def get_classifications(self, x):
        norms = torch.nn.functional.softmax(x, dim=1)
        masks = norms.argmax(1).float()
        return torch.unsqueeze(masks, 1)

    
    def forward(self, x):
        return self.network.predict(x)