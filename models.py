import networks
import net_gen
import torch
from torch import nn
import pytorch_lightning as pl
import torchvision as tv

def save_grid(tb, inp, name, epoch):  
    img_grid = tv.utils.make_grid(inp, normalize=True, scale_each=True)
    tb.add_image(name, img_grid, epoch)

def get_classifications(x):
    norms = torch.nn.functional.softmax(x, dim=1)
    masks = norms.argmax(1).float()

    # masks = torch.unsqueeze(masks, 1) * 255.0/9
    # masks = torch.cat((masks, masks, masks), dim=1)
    return masks

class DenseSimSiam(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        #self.encoder = net_gen.ResnetEncoder(kwargs["num_channels"], 512, use_dropout=True)
        self.encoder = net_gen.ResnetGenerator(kwargs["num_channels"], 10)
        self.projector_1 = networks.DenseProjector(num_channels=10)
        self.predictor_1 = networks.DensePredictor(num_classes=10)
        self.projector_2 = networks.DenseProjectorMLP()
        self.predictor_2 = networks.DensePredictorMLP()
        self.loss = nn.CrossEntropyLoss()
        self.softmax = torch.nn.Softmax(dim=1)
        self.logmax = torch.nn.LogSoftmax(dim=1)

    def training_step(self, x, idx):
        viz, inp, inp_aug = x["viz"].squeeze(), x["base"].squeeze(), x["rand"].squeeze()

        x1 = self.encoder(inp)
        x2 = self.encoder(inp_aug)

        z1 = self.projector_1(x1)
        z2 = self.projector_1(x2)

        p1 = self.predictor_1(z1)
        p2 = self.predictor_1(z2)

        z1_stop = z1.detach()
        z2_stop = z2.detach()

        e1 = torch.matmul(x1, z1).flatten(start_dim=2)
        e2 = torch.matmul(x2, z2).flatten(start_dim=2)


        v1 = self.projector_2(e1)
        u1 = self.predictor_2(v1)

        v2 = self.projector_2(e2)

        self.log_images(viz,z1,p1)

        pix_loss = (-(self.softmax(p1).mean() * self.logmax(z2_stop).mean())-(self.softmax(p2).mean() * self.logmax(z1_stop).mean())) * 0.5
        region_loss = self.loss(u1, v2)
        loss = (pix_loss + region_loss) * 0.5
        return loss

    def log_images(self, viz, proj, pred):
        tb = self.logger.experiment
        sample_imgs = viz[:6]
        save_grid(tb, sample_imgs, 'rgb', self.current_epoch)
           
        sample_proj = proj[:6]
        sample_proj = get_classifications(sample_proj)
        save_grid(tb, sample_proj, 'projected', self.current_epoch)

        sample_pred = pred[:6]
        sample_pred = get_classifications(sample_pred)
        save_grid(tb, sample_proj, 'predicted', self.current_epoch)


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=5e-3)
        return optimizer
    
    def grid_sample(self, x):
        return None


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
        #self.loss = nn.NLLLoss2d()
        #self.loss=nn.BCEWithLogitsLoss()

    #TODO: Integrate LR with KWARGS
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=5e-3)
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

        #loss = (self.loss(p1, z2).mean() + self.loss(p2, z1).mean()) *0.5
        loss = (self.loss(p1,z2.softmax(dim=1)) + self.loss(p2, z1.softmax(dim=1)))*.05
        self.log('train_loss', loss)
        return loss
    
    def save_grid(self, inp, name):
        tb = self.logger.experiment
        img_grid = tv.utils.make_grid(inp, normalize=True, scale_each=True)
        tb.add_image(name, img_grid, self.current_epoch)



    # TODO: Need species info to do validation
    # def validation_step(self, x, batch_idx):
    #     return None

    #TODO: make 3 channel color image
    def get_classifications(self, x):
        norms = torch.nn.functional.softmax(x, dim=1)
        masks = norms.argmax(1).float()

        masks = torch.unsqueeze(masks, 1) * 255.0/9
        masks = torch.cat((masks, masks, masks), dim=1)
        return masks



    
    def forward(self, x):
        return self.network.predict(x)