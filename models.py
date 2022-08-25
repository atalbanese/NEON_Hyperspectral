from requests import patch
import networks
import net_gen
import torch
from torch import layer_norm, nn, einsum
import torch.nn.functional as f
import torchvision.transforms.functional as TF
import pytorch_lightning as pl
import torchvision as tv
import transforms as tr
import torch.hub
from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange
#import cv2 as cv
import numpy.ma as ma
import numpy as np
from attrs import define, field
from dall_e import Encoder, Decoder
import matplotlib.pyplot as plt


#TODO: test out additional augmentations for labelled training
class SwaVModelUnified(pl.LightningModule):
    def __init__(self, 
                class_key,
                lr, 
                class_weights, 
                features_dict, 
                num_intermediate_classes,
                pre_training,
                mode='default', 
                augment_refine = 'false',
                scheduler=True,
                positions=False,
                emb_size=256,
                augment_bright=False,
                filters=[]):
        super().__init__()
        self.save_hyperparameters()
        self.features_dict = features_dict
        self.filters=filters
        self.num_channels = self.calc_num_channels()
        self.num_output_classes = len(class_key.keys())
        self.missing_data = nn.Parameter(torch.randn((1)))
        self.lr = lr
        self.augment_refine = augment_refine
        self.augment_bright = augment_bright
        if self.augment_bright:
            self.ab = tr.BrightnessAugment(p=1)
        self.mode = mode
        self.scheduler = scheduler
        if self.mode == 'default':
            self.swav = networks.SWaVUnified(self.num_channels, num_intermediate_classes, n_head=8, n_layers=8, positions=positions)
        if self.mode == 'patch':
            self.swav = networks.SWaVUnifiedPerPatch(self.num_channels, num_intermediate_classes, n_head=8, n_layers=8, patch_size=4, positions=positions, emb_size=emb_size)
        if self.mode == 'pixel_patch':
            self.swav = networks.SWaVUnifiedPerPixelPatch(self.num_channels, num_intermediate_classes)
        if self.mode== 'resnet':
            self.swav = tv.models.segmentation.fcn_resnet50(pretrained_backbone=False, num_classes=4)
        if self.mode == 'default':
            self.predict =  nn.Sequential(nn.Linear(num_intermediate_classes, num_intermediate_classes*2),
                                         nn.BatchNorm1d(num_intermediate_classes*2),
                                         nn.ReLU(),
                                         nn.Linear(num_intermediate_classes*2, self.num_output_classes))
        else:
            self.predict = networks.UpsamplePredictor(num_intermediate_classes, self.num_output_classes)
        if class_weights is not None:
            self.loss = nn.CrossEntropyLoss(weight=torch.tensor(class_weights), label_smoothing=0.2, ignore_index=-1)
        else:
            self.loss = nn.CrossEntropyLoss(label_smoothing=0.2, ignore_index=-1)
            # self.loss = torch.hub.load(
            # 'adeelh/pytorch-multi-class-focal-loss',
            # model='FocalLoss',
            # gamma=2,
            # reduction='mean',
            # force_reload=False,
            # ignore_index=-1
            # )

        self.class_key = {value:key for key, value in class_key.items()}
        self.results = self.make_results_dict(self.class_key)
        self.pre_training = pre_training
        self.ra = Rearrange('b c h w -> b (h w) c')


    def calc_num_channels(self):
        out = 0
        for v in self.features_dict.values():
            out += v
        return out

    def make_results_dict(self, classes):
        out = {value:{v:0 for v in classes.values()} for value in classes.values()}
        return out

    def update_results(self, inp, target):
        
        #target = torch.argmax(target, dim=1)
        pred = list(inp)
        targets = list(target)

        for p, t in list(zip(pred, targets)):
            if (int(t)) != -1:
                p_label = self.class_key[int(p)]
                t_label = self.class_key[int(t)]

                self.results[t_label][p_label] += 1
        return None

    def calc_ova(self):
        total_pixels = 0
        accurate_pixels = 0
        for key, value in self.results.items():
            for i, j in value.items():
                total_pixels += j
                if key == i:
                    accurate_pixels += j

        if total_pixels != 0:            
            return accurate_pixels/total_pixels
        return 0

    def prep_data(self, inp):
        to_cat = []
        for key, value in self.features_dict.items():
            if key == 'mask':
                continue
            if key == 'targets':
                continue
            if key == 'height':
                continue
            cur = inp[key]
            if len(cur.shape) != 4:
                cur = cur.unsqueeze(1)
            if cur.shape[1] != value:
                cur = cur[:,:value,...]
            to_cat.append(cur.float())
        return torch.cat(to_cat, dim=1)

    def pre_training_step(self, inp):
        inp = self.prep_data(inp)
        if self.mode != 'resnet':
            inp[inp != inp] = self.swav.transforms_main.missing
        else:
            inp[inp != inp] = self.missing_data

        if torch.rand(1) > 0.5:
            inp = TF.vflip(inp)

        if torch.rand(1) > 0.5:
            inp = TF.hflip(inp)

        if self.mode == 'default':
            inp = self.ra(inp)

        loss = self.swav.forward_train(inp)
        self.log('pre_train_loss', loss)
        return loss

    def refine_step(self, inp, validating=False):
        targets = inp['targets']
        mask = inp['mask']
        for f in self.filters:
            to_mask = mask[f]
            #to_mask[to_mask != to_mask]
            targets[to_mask] = -1
        crop_coords = None
        if 'crop_coords' in inp.keys():
            crop_coords = inp['crop_coords']
        inp = self.prep_data(inp)



        if self.mode == 'default':
            inp = self.ra(inp)
        if not validating:
            if self.augment_refine:
                inp = self.swav.transforms_main(inp)
            if self.augment_bright:
                inp = self.ab(inp)
            if torch.rand(1) > 0.5:
                inp = TF.vflip(inp)
                
                targets=TF.vflip(targets)

            if torch.rand(1) > 0.5:
                inp = TF.hflip(inp)

                targets=TF.hflip(targets)
        
        if self.mode != 'resnet':
            inp[inp != inp] = self.swav.transforms_main.missing
        else:
            inp[inp != inp] = self.missing_data
            
        inp = self.swav.forward(inp)
        if self.mode == 'default':
            inp = rearrange(inp, 'b s f -> (b s) f')
            inp = self.predict(inp)
        elif self.mode == 'patch':
            inp = rearrange(inp, 'b (h w) f -> b f h w', h=5, w=5)
            inp = self.predict(inp)

        elif self.mode == 'resnet':
            inp = inp['out']

        inp = torch.softmax(inp, dim=1)

        
        if crop_coords is None:
            targets = targets.to(torch.long)
        else:
            targets=torch.argmax(targets, dim=1)
        loss = self.loss(inp, targets)

        inp = torch.argmax(inp, dim=1)

        

        if not validating: 
            self.log('train_loss', loss)
            return loss
        else:
            self.log('valid_loss', loss)

            inp = inp.flatten()
            targets = targets.flatten()
            self.update_results(inp, targets)
            return loss



    def training_step(self, inp):
        
        if self.pre_training:
            return self.pre_training_step(inp)
            
        else:
            return self.refine_step(inp)

        return None

    def validation_step(self, inp, _):
        if not self.pre_training:
            return self.refine_step(inp, validating=True)
        return None

    def validation_epoch_end(self, _):
        if not self.pre_training:
            ova = self.calc_ova()
            print(self.results)

            self.results = self.make_results_dict(self.class_key)
            self.log('ova', ova)

            print(f'\nOVA: {ova}')
        

    def test_step(self, x, _):
        if not self.pre_training:
            self.validation_step(x, _)

        return None

    def test_epoch_end(self, _):
        if not self.pre_training:
            ova = self.calc_ova()
            print(self.results)

            self.results = self.make_results_dict(self.class_key)
            self.log('test_ova', ova)

            print(f'\nOVA: {ova}')
            return None

    def on_before_optimizer_step(self, optimizer, optimizer_idx):
        if (self.current_epoch < 1) and (self.pre_training):
            for name, p in self.swav.named_parameters():
                    if "prototypes" in name:
                        p.grad = None
    
    def on_train_batch_start(self, batch, batch_idx):
        if self.pre_training:
            self.swav.norm_prototypes()

    def configure_optimizers(self):
        if self.pre_training:
            optimizer = torch.optim.AdamW(self.swav.parameters(), lr=self.lr)
        else:
            optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()), lr=self.lr)
        
        to_return = {"optimizer": optimizer, "monitor": "train_loss"}
        if self.scheduler:

            #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 20, eta_min=0)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer, T_0=40)
            to_return['lr_scheduler'] = scheduler
        return to_return
    
    
    def forward(self, x):
        x = self.prep_data(x)
        out = self.swav.forward(x)
        out = rearrange(out, 'b s f -> (b s) f')
        out = self.predict(out)
        return out
    
    

    



#TODO: Make checkpoint saving work
class SWaVModelRefine(pl.LightningModule):
    def __init__(self, swav_config, num_output_classes, class_key, chm_mean, chm_std, lr=5e-4, height_threshold=5, class_weights=None, freeze_backbone=True, trained_backbone=True,**kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.freeze_backbone = freeze_backbone
        self.height_threshold = height_threshold
        self.lr = lr
        self.chm_mean = chm_mean
        self.chm_std = chm_std
        if trained_backbone:
            if freeze_backbone:
                self.model = SWaVModelSuperPixel(**swav_config).load_from_checkpoint(swav_config['ckpt'],**swav_config).eval()
            else:
                self.model = SWaVModelSuperPixel(**swav_config).load_from_checkpoint(swav_config['ckpt'],**swav_config)
        else:
            self.freeze_backbone = False
            self.model = SWaVModelSuperPixel(**swav_config)

        self.predict_mlp = nn.Sequential(nn.Linear(swav_config['num_classes'], swav_config['num_classes']*2),
                                         nn.BatchNorm1d(swav_config['num_classes']*2),
                                         nn.ReLU(),
                                         nn.Linear(swav_config['num_classes']*2, num_output_classes))
        #self.positions = nn.Parameter(torch.randn((31, 4, 4)))
        if class_weights is not None: 
            self.loss = nn.CrossEntropyLoss(weight=torch.tensor(class_weights))
        else:
            self.loss = nn.CrossEntropyLoss()

        self.results = self.make_results_dict(class_key)
        self.class_key = class_key
        self.ra = Rearrange('b c h w -> b (h w) c')




    def make_results_dict(self, classes):
        out = {value:{v:0 for v in classes.values()} for value in classes.values()}
        return out

    def update_results(self, inp, target):
        inp = torch.argmax(inp, dim=1)
        pred = list(inp)
        targets = list(target)

        for p, t in list(zip(pred, targets)):
            p_label = self.class_key[int(p)]
            t_label = self.class_key[int(t)]

            self.results[t_label][p_label] += 1

        return None

    
    def calc_ova(self):
        total_pixels = 0
        accurate_pixels = 0
        for key, value in self.results.items():
            for i, j in value.items():
                total_pixels += j
                if key == i:
                    accurate_pixels += j

        if total_pixels != 0:            
            return accurate_pixels/total_pixels
        return 0



    def training_step(self, x):
        pca = x['pca']
        ica = x['ica']
        raw = x['raw_bands']
        shadow = x['shadow'].unsqueeze(1)
        chm = x['chm'].unsqueeze(1)
        az = x['azm'].unsqueeze(1)
        targets = x['target']
        height = x['height']
        mask = x['mask']
        mask = reduce(mask, 'b c h w -> b () h w', 'max')

        height_mask = torch.zeros(chm.shape, dtype=torch.bool, device=torch.device('cuda'))

        for i, h in enumerate(height):
            height_test = chm[i]
            add_mask = height_test < (h - self.height_threshold)
            height_mask[i] = add_mask

        chm = (chm - self.chm_mean)/self.chm_std
        inp = torch.cat((pca,ica,raw,shadow,chm,az), dim=1)

        mask += height_mask
        mask = ~mask

        if torch.rand(1) > 0.5:
            inp = TF.vflip(inp)
            mask = TF.vflip(mask)
            targets=TF.vflip(targets)

        if torch.rand(1) > 0.5:
            inp = TF.hflip(inp)
            mask = TF.hflip(mask)
            targets=TF.hflip(targets)

        mask = rearrange(mask, 'b c h w -> (b h w) c').squeeze()
        #inp += self.positions


        inp = self.ra(inp)
        if self.freeze_backbone:
            with torch.no_grad():
                inp = self.model.forward(inp)
            inp = rearrange(inp, 'b s f -> (b s) f')
            inp.requires_grad = True
        else:
            inp = self.model.forward(inp)
            inp = rearrange(inp, 'b s f -> (b s) f')
        inp = self.predict_mlp(inp)

        #Try without softmax
        inp = torch.softmax(inp, dim=1)
        inp = inp[mask, :]

        targets= rearrange(targets, 'b c h w -> (b h w) c')
        targets=torch.argmax(targets, dim=1)
        targets = targets[mask]


        loss = self.loss(inp, targets)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, x, _):
        pca = x['pca']
        ica = x['ica']
        raw = x['raw_bands']
        shadow = x['shadow'].unsqueeze(1)
        chm = x['chm'].unsqueeze(1)
        az = x['azm'].unsqueeze(1)
        targets = x['target']
        height = x['height']
        mask = x['mask']
        mask = reduce(mask, 'b c h w -> b () h w', 'max')

        height_mask = torch.zeros(chm.shape, dtype=torch.bool, device=torch.device('cuda'))

        for i, h in enumerate(height):
            height_test = chm[i]
            add_mask = height_test < (h - self.height_threshold)
            height_mask[i] = add_mask

        chm = (chm - self.chm_mean)/self.chm_std
        inp = torch.cat((pca,ica,raw,shadow,chm,az), dim=1)
        #inp += self.positions

        mask += height_mask
        mask = ~mask
        mask = rearrange(mask, 'b c h w -> (b h w) c').squeeze()

        
        inp = self.ra(inp)
        inp = self.forward(inp)
        inp = torch.softmax(inp, dim=1)
        inp = inp[mask, :]

        targets= rearrange(targets, 'b c h w -> (b h w) c')
        targets=torch.argmax(targets, dim=1)
        targets = targets[mask]

        self.update_results(inp, targets)

        return None

    def validation_epoch_end(self, _):
        ova = self.calc_ova()
        print(self.results)

        self.results = self.make_results_dict(self.class_key)
        self.log('ova', ova)

        print(f'\nOVA: {ova}')
        

    def test_step(self, x, _):
        self.validation_step(x, _)

        return None

    def test_epoch_end(self, _):
        ova = self.calc_ova()
        print(self.results)

        self.results = self.make_results_dict(self.class_key)
        self.log('ova', ova)

        print(f'\nOVA: {ova}')
        return None

    




    def configure_optimizers(self):
        if self.freeze_backbone:
            optimizer = torch.optim.AdamW(self.predict_mlp.parameters(), lr=self.lr)
        else:
             optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 20, eta_min=0)
        #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        #scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 20, T_mult=2)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "train_loss"}
    
    
    def forward(self, x):
        out = self.model.forward(x)
        out = rearrange(out, 'b s f -> (b s) f')
        out = self.predict_mlp(out)
        return out

class SWaVModelSuperPixel(pl.LightningModule):
    def __init__(self, 
                    azm=True,
                    chm=True, 
                    pop_queue_start=14, 
                    queue_start=15, 
                    use_queue=False,  
                    queue_chunks=1, 
                    num_classes=12, 
                    azm_concat=False, 
                    chm_concat=False,
                    positions=False, 
                    **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.model = networks.SWaVSuperPixel(azm=azm, 
                                                chm=chm, 
                                                queue_chunks=queue_chunks, 
                                                num_classes=num_classes, 
                                                azm_concat=azm_concat, 
                                                chm_concat=chm_concat,
                                                positions=positions)
        self.use_queue = use_queue
        self.pop_queue_start = pop_queue_start
        self.queue_start = queue_start
        self.ra = Rearrange('b c h w -> b (h w) c')


    def training_step(self, x):
        pca = x['pca']
        ica = x['ica']
        raw = x['raw_bands']
        shadow = x['shadow'].unsqueeze(1)
        chm = x['chm'].unsqueeze(1)
        az = x['azm'].unsqueeze(1)


        inp = torch.cat((pca,ica,raw,shadow,chm,az), dim=1)

        if torch.rand(1) > 0.5:
            inp = TF.vflip(inp)

        if torch.rand(1) > 0.5:
            inp = TF.hflip(inp)

        inp = self.ra(inp)

        loss = self.model.forward_train(inp)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 20, eta_min=5e-9)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "train_loss"}

    def on_before_optimizer_step(self, optimizer, optimizer_idx):
        if self.current_epoch < 1:
            for name, p in self.model.named_parameters():
                    if "prototypes" in name:
                        p.grad = None
    
    def on_train_batch_start(self, batch, batch_idx):
        self.model.norm_prototypes()
        if self.use_queue:
            if self.current_epoch == self.pop_queue_start:
                self.model.populate_queue = True
            if self.current_epoch == self.queue_start:
                self.model.use_queue = True

    def forward(self, x):

        return self.model(x)



#TODO: On train end, log results and run validation
class SWaVModelStruct(pl.LightningModule):
    def __init__(self, patch_size, img_size, azm=True, chm=True, pop_queue_start=14, queue_start=15, use_queue=False,  same_embed=False, concat=False, queue_chunks=1, num_classes=12, azm_concat=False, chm_concat=False, aug_brightness=False, main_brightness=False, **kwargs):
        super().__init__()
        self.model = networks.SWaVStruct(patch_size=patch_size, img_size=img_size, azm=azm, chm=chm, same_embed=same_embed, concat=concat, queue_chunks=queue_chunks, num_classes=num_classes, azm_concat=azm_concat, chm_concat=chm_concat, aug_brightness=aug_brightness)
        self.img_size = img_size
        self.use_queue = use_queue
        self.pop_queue_start = pop_queue_start
        self.queue_start = queue_start
        if main_brightness:
            self.brightness = tr.BrightnessAugment()
        else:
            self.brightness = None
        #self.chm_embed = nn.Conv2d(1, 1, kernel_size=patch_size, stride=1)
        #self.azm_embed = nn.Conv2d(1, 1, kernel_size=patch_size, stride=1)

    def training_step(self, x):
        inp = x['base'].squeeze(0)
        chm = x['chm'].squeeze(0).unsqueeze(1)
        az = x['azimuth'].squeeze(0).unsqueeze(1)

        # inp = TF.center_crop(inp, self.img_size)
        # chm = TF.center_crop(chm, self.img_size)
        # az = TF.center_crop(az, self.img_size)

        if self.brightness is not None:
            inp = self.brightness(inp)

        if torch.rand(1) > 0.5:
            inp = TF.vflip(inp)
            chm = TF.vflip(chm)
            az = TF.vflip(az)

        if torch.rand(1) > 0.5:
            inp = TF.hflip(inp)
            chm = TF.hflip(chm)
            az = TF.hflip(az)


        loss = self.model.forward_train(inp, chm, az)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 10, eta_min=5e-7)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "train_loss"}

    def on_before_optimizer_step(self, optimizer, optimizer_idx):
        if self.current_epoch < 1:
            for name, p in self.model.named_parameters():
                    if "prototypes" in name:
                        p.grad = None
    
    def on_train_batch_start(self, batch, batch_idx):
        self.model.norm_prototypes()
        if self.use_queue:
            if self.current_epoch == self.pop_queue_start:
                self.model.populate_queue = True
            if self.current_epoch == self.queue_start:
                self.model.use_queue = True

    def forward(self, x, chm, azm):
        return self.model(x, chm, azm)


class SWaVModelRes(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = networks.SWaVResnet()

    def training_step(self, x):
        inp = x['base'].squeeze(0)
        loss = self.model.forward_train(inp)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 10, eta_min=5e-7)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "train_loss"}

    def on_before_optimizer_step(self, optimizer, optimizer_idx):
        if self.current_epoch < 1:
            for name, p in self.model.named_parameters():
                    if "prototypes" in name:
                        p.grad = None
    
    def on_train_batch_start(self, batch, batch_idx):
        self.model.norm_prototypes()

    def forward(self, x):
        return self.model(x)

class SWaVModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = networks.SWaV(patch_size=4)

    def training_step(self, x):
        inp = x['base'].squeeze(0)
        loss = self.model.forward_train(inp)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 10, eta_min=5e-7)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "train_loss"}

    def on_before_optimizer_step(self, optimizer, optimizer_idx):
        if self.current_epoch < 1:
            for name, p in self.model.named_parameters():
                    if "prototypes" in name:
                        p.grad = None
    
    def on_train_batch_start(self, batch, batch_idx):
        self.model.norm_prototypes()

    def forward(self, x):
        return self.model(x)


class TransEmbedConvSimSiam(pl.LightningModule):
    def __init__(self, num_channels, batch_size = 512, emb_size=1024, img_size=256, output_classes=256):
        super().__init__()
        self.batch_size = batch_size
        self.img_size = img_size
        self.output_classes = output_classes

        #self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        
        encoder_layer = torch.nn.TransformerEncoderLayer(d_model=num_channels, nhead=num_channels//5, dim_feedforward=num_channels*2, batch_first=True)
        self.embed = torch.nn.TransformerEncoder(encoder_layer, num_layers=4)

        self.flatten = Rearrange('b c h w -> b (h w) c')
        self.unflatten = Rearrange('b (h w) c -> b c h w', h=self.img_size, w=self.img_size)

        self.down_channel = nn.Sequential(nn.Linear(num_channels, num_channels//2),
                                          nn.BatchNorm1d(img_size**2),
                                          nn.ReLU(),
                                          nn.Linear(num_channels//2, 3))
        
        self.encoder = net_gen.ResnetGenerator(3,3, norm_layer=nn.InstanceNorm2d, use_dropout=True, n_blocks=18)
        
       
        self.proj = nn.Sequential(nn.Conv2d(3, 3, kernel_size=1),
                                  nn.BatchNorm2d(3),
                                  nn.ReLU(),
                                  nn.Conv2d(3, 9, kernel_size=3, padding=1),
                                  nn.BatchNorm2d(9),
                                  nn.ReLU(),
                                  nn.Conv2d(9, 20, kernel_size=3, padding=1))
        #self.fusion = networks.Fusion(num_channels=output_classes)

        self.pred = nn.Sequential(nn.Conv2d(20, 20, kernel_size=3, padding=1),
                                  nn.BatchNorm2d(20, affine=False),
                                  nn.ReLU(),
                                  nn.Conv2d(20, 20, kernel_size=3, padding=1, bias=False))

        self.softmax = nn.Softmax(dim=1)
        self.logmax = nn.LogSoftmax(dim=1)
        self.loss = networks.VitSiamLoss()
 

            
    def training_step(self, x):
        #Mask is True where NANs existed before so we want to invert it
        inp, inp_aug, mask = x['base'].squeeze(0), x['augment'].squeeze(0), ~x['mask'].squeeze(0)
        mask = mask[:,0,:,:]
        mask = mask.unsqueeze(dim=1)

        mask = repeat(mask, 'b c h w -> b (repeat c) h w', repeat=self.output_classes)

        z_1 = self.flatten(inp)
        z_2 = self.flatten(inp_aug)
        
        z_1 = self.embed(z_1)
        z_2 = self.embed(z_2)

        z_1 = self.down_channel(z_1)
        z_2 = self.down_channel(z_2)

        z_1 = self.unflatten(z_1)
        z_2 = self.unflatten(z_2)

        z_1 = self.encoder(z_1)
        z_2 = self.encoder(z_2)

        z_1 = self.proj(z_1)
        z_2 = self.proj(z_2)

        p_1 = self.pred(z_1)
        p_2 = self.pred(z_2)

        


        z_1, z_2, = z_1.detach(), z_2.detach()

        loss = (self.loss(p_1[mask], z_2[mask]) + self.loss(p_2[mask], z_1[mask])) *0.5

        self.log('train_loss', loss)

        return loss/8

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=5e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 10)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "train_loss"}

    def depatch(self, x):
        x = rearrange(x, 'b (h w) c -> b c h w', h=self.num_patches, w=self.num_patches)
        #x = f.interpolate(x, scale_factor=2)
        return x

    def forward(self, x):
        z_1 = self.flatten(x)
       
        #z_1 = self.embed(z_1)
       
        z_1 = self.down_channel(z_1)
   
        z_1 = self.unflatten(z_1)

        z_1 = self.encoder(z_1)
 
        z_1 = self.proj(z_1)
 
        p_1 = self.pred(z_1)
        x = torch.argmax(p_1, dim=1)
        return x


class PatchedSimSiam(pl.LightningModule):
    def __init__(self, num_channels, batch_size = 512, emb_size=1024, img_size=256, patch_size=8, output_classes=256):
        super().__init__()
        self.batch_size = batch_size
        self.img_size = img_size
        self.output_classes = output_classes
        self.patch_size = patch_size
        self.num_patches = self.img_size//self.patch_size

        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

        self.embed = networks.LinearPatchEmbed(in_channels=num_channels, patch_size=patch_size, emb_size=emb_size, img_size=img_size)
        encoder_layer = torch.nn.TransformerEncoderLayer(d_model=emb_size, nhead=8, dim_feedforward=1024, batch_first=True)
        self.t_enc = torch.nn.TransformerEncoder(encoder_layer, num_layers=4)
        self.map = networks.MapToken(emb_size=emb_size, patches=img_size**2//(patch_size**2))
       
        self.proj = networks.DenseProjectorMLP(num_channels=emb_size)
        #self.fusion = networks.Fusion(num_channels=output_classes)

        self.pred = networks.DenseVitPredict(output_classes)

        self.softmax = nn.Softmax(dim=1)
        self.logmax = nn.LogSoftmax(dim=1)
        self.loss = networks.VitSiamLoss()
 

            
    def training_step(self, x):
        #Mask is True where NANs existed before so we want to invert it
        inp, inp_aug, mask = x['base'].squeeze(0), x['augment'].squeeze(0), ~x['mask'].squeeze(0)
        mask = mask[:,0,:,:]
        mask = mask.unsqueeze(dim=1)

        mask = repeat(mask, 'b c h w -> b (repeat c) h w', repeat=self.output_classes)

        e_1 = self.embed(inp)

        e_2 = self.embed(inp_aug)

        z_1 = self.t_enc(e_1)
        z_1 = self.map(z_1)
        z_1 = self.proj(z_1)
        z_1 = self.depatch(z_1)
        z_1 = self.up(z_1)
        z_1 = self.up(z_1)

        z_2 = self.t_enc(e_2)
        z_2 = self.map(z_2)
        z_2 = self.proj(z_2)
        z_2 = self.depatch(z_2)
        z_2 = self.up(z_2)
        z_2 = self.up(z_2)


        p_1 = self.pred(z_1)
        p_2 = self.pred(z_2)

        # inp = inp.detach()
        # inp = inp[0]
        # inp = inp[0:3]
        # self.logger.experiment.add_image('input', inp, self.current_epoch, dataformats='CHW')

        # to_viz = z_1.detach()[0]
        # to_viz = self.up(to_viz.unsqueeze(0)).squeeze(0)
        # to_viz = torch.argmax(to_viz, dim=0)
        # save_grid(self.logger.experiment, to_viz, 'projected', self.current_epoch)

        # to_viz = p_1.detach()[0]
        # to_viz = self.up(to_viz.unsqueeze(0)).squeeze(0)
        # to_viz = torch.argmax(to_viz, dim=0)
        # save_grid(self.logger.experiment, to_viz, 'predicted', self.current_epoch)

        z_1, z_2, = z_1.detach(), z_2.detach()

        loss = (self.loss(p_1[mask], z_2[mask]) + self.loss(p_2[mask], z_1[mask])) *0.5

        self.log('train_loss', loss)

        return loss/8

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=5e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 10)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "train_loss"}

    def depatch(self, x):
        x = rearrange(x, 'b (h w) c -> b c h w', h=self.num_patches, w=self.num_patches)
        #x = f.interpolate(x, scale_factor=2)
        return x

    def forward(self, x):
        x = self.embed(x)
        x = self.t_enc(x)
        x = self.map(x)
        
        x = self.proj(x)
        x = self.depatch(x)
        x = self.up(x)
        x = self.up(x)
        x = self.pred(x)
        x = torch.argmax(x, dim=1)
        return x
        



class MaskedVitSiam(pl.LightningModule):
    def __init__(self, num_channels, batch_size = 512, emb_size=1024, img_size=256, patch_size=8, output_classes=256):
        super().__init__()
        self.batch_size = batch_size
        self.img_size = img_size
        self.output_classes = output_classes
        self.patch_size = patch_size
        self.num_patches = self.img_size//self.patch_size

        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

        self.embed = networks.LinearPatchEmbed(in_channels=num_channels, patch_size=patch_size, emb_size=emb_size, img_size=img_size)
        #self.embed = networks.ResnetPatchEmbed(in_channels=num_channels, patch_size=patch_size, emb_size=emb_size, img_size=img_size)

        encoder_layer = torch.nn.TransformerEncoderLayer(d_model=emb_size, nhead=8, dim_feedforward=1024, batch_first=True)
        self.t_enc = torch.nn.TransformerEncoder(encoder_layer, num_layers=4)
        self.map = networks.MapToken(emb_size=emb_size, patches=img_size**2//(patch_size**2))
        #self.proj = networks.VitProject(output_classes, emb_size=emb_size)
        #Might want to use Group or Layer Norm
        # self.resample32 = nn.Sequential(nn.Conv2d(emb_size, output_classes, kernel_size=1),
        #                                 nn.BatchNorm2d(output_classes),
        #                                 nn.ReLU(),
        #                                 nn.Conv2d(output_classes, output_classes, kernel_size=3, stride=4))

        # self.resample16 = nn.Sequential(nn.Conv2d(emb_size, output_classes, kernel_size=1),
        #                                 nn.BatchNorm2d(output_classes),
        #                                 nn.ReLU(),
        #                                 nn.Conv2d(output_classes, output_classes, kernel_size=3, stride=2, padding=1))
        # self.resample8 = nn.Sequential(nn.Conv2d(emb_size, output_classes, kernel_size=1),
        #                                 nn.BatchNorm2d(output_classes),
        #                                 nn.ReLU(),
        #                                 nn.Conv2d(output_classes, output_classes, kernel_size=3, stride=1, padding=1))
        # self.resample4 = nn.Sequential(nn.Conv2d(emb_size, output_classes, kernel_size=1),
        #                                nn.BatchNorm2d(output_classes),
        #                                 nn.ReLU(),
        #                                 nn.ConvTranspose2d(output_classes, output_classes, kernel_size=3, stride=2, padding=1),
        #                                 #nn.Upsample((img_size//4,img_size//4))
        #                                 )

        self.resample32 = nn.Sequential(nn.Conv2d(emb_size, output_classes, kernel_size=1),
                                        nn.BatchNorm2d(output_classes),
                                        nn.ReLU(),
                                        nn.Conv2d(output_classes, output_classes, kernel_size=3, stride=8),
                                        )

        self.resample16 = nn.Sequential(nn.Conv2d(emb_size, output_classes, kernel_size=1),
                                        nn.BatchNorm2d(output_classes),
                                        nn.ReLU(),
                                        nn.Conv2d(output_classes, output_classes, kernel_size=3, stride=4, padding=1))
        self.resample8 = nn.Sequential(nn.Conv2d(emb_size, output_classes, kernel_size=1),
                                        nn.BatchNorm2d(output_classes),
                                        nn.ReLU(),
                                        nn.Conv2d(output_classes, output_classes, kernel_size=3, stride=2, padding=1))
        self.resample4 = nn.Sequential(nn.Conv2d(emb_size, output_classes, kernel_size=1),
                                       nn.BatchNorm2d(output_classes),
                                        nn.ReLU(),
                                        nn.Conv2d(output_classes, output_classes, kernel_size=3, stride=1, padding=1),
                                        #nn.Upsample((img_size//4,img_size//4))
                                        )

        self.resamplers = [self.resample4, self.resample8, self.resample16, self.resample32]

        # self.fusion32 = networks.Fusion(num_channels=output_classes)
        # self.fusion16 = networks.Fusion(num_channels=output_classes)
        # self.fusion8 = networks.Fusion(num_channels=output_classes)
        # self.fusion4 = networks.Fusion(num_channels=output_classes)

        self.fusion = networks.Fusion(num_channels=output_classes)

        self.pred = networks.DenseVitPredict(output_classes)

        self.softmax = nn.Softmax(dim=1)
        self.logmax = nn.LogSoftmax(dim=1)
        self.loss = networks.VitSiamLoss()
        #self.loss = nn.CrossEntropyLoss()
        self.features = {}
        
        self.get_layers = [0,1,2,3]
        self.register_hooks()

    def register_hooks(self):
        for to_hook in self.get_layers:
            self.t_enc.layers[to_hook].register_forward_hook(self.get_features(to_hook, self.features))

    @staticmethod
    def get_features(name, features):
        def hook(model, input, output):
            features[name] = output
        return hook

    def reassamble_layers(self):
        for ix, layer in enumerate(self.get_layers):
            feature = self.features[layer]
            feature = self.map(feature)
            feature = self.depatch(feature)
            feature = self.resamplers[ix](feature)
            self.features[layer] = feature

    # def fuse_layers(self):
    #     a = self.fusion32(self.features[11])
    #     b = self.fusion16(self.features[8], z=a)
    #     c = self.fusion8(self.features[5], z=b)
    #     d = self.fusion4(self.features[2], z=c)
    #     return d

    def fuse_layers(self):
        a = self.fusion(self.features[self.get_layers[3]])
        b = self.fusion(self.features[self.get_layers[2]], z=a)
        c = self.fusion(self.features[self.get_layers[1]], z=b)
        d = self.fusion(self.features[self.get_layers[0]], z=c)
        return d

    

            

    def training_step(self, x):
        #Mask is True where NANs existed before so we want to invert it
        inp, inp_aug, mask = x['base'].squeeze(0), x['augment'].squeeze(0), ~x['mask'].squeeze(0)
        mask = mask[:,0,:,:]
        mask = mask.unsqueeze(dim=1)

        mask = repeat(mask, 'b c h w -> b (repeat c) h w', repeat=self.output_classes)

        e_1 = self.embed(inp)

        e_2 = self.embed(inp_aug)

        self.t_enc(e_1)
        self.reassamble_layers()
        z_1 = self.fuse_layers()
        z_1 = self.up(z_1)

        # e_1 = self.map(e_1)
        # e_1 = self.depatch(e_1)
        # e_1 = self.up(e_1)
        # e_1 = self.up(e_1)
        # e_1 = e_1.detach()
        # e_1 = e_1[0]

        # to_viz = torch.argmax(e_1, dim=0)
        # save_grid(self.logger.experiment, to_viz, 'embed', self.current_epoch)

        # e_1 = self.resample4(e_1)
        # e_1 = self.fusion(e_1)
        # e_1 = self.fusion(e_1)

        
        # z_1 = z_1 + e_1


        self.t_enc(e_2)
        self.reassamble_layers()
        z_2 = self.fuse_layers()
        z_2 = self.up(z_2)

        # e_2 = self.map(e_2)
        # e_2 = self.depatch(e_2)
        # e_2 = self.resample4(e_2)
        # e_2 = self.fusion(e_2)
        # e_2 = self.fusion(e_2)

        # z_2 = z_2 + e_2


        p_1 = self.pred(z_1)
        p_2 = self.pred(z_2)

        inp = inp.detach()
        inp = inp[0]
        inp = inp[0:3]
        self.logger.experiment.add_image('input', inp, self.current_epoch, dataformats='CHW')

        to_viz = z_1.detach()[0]
        to_viz = torch.argmax(to_viz, dim=0)
        save_grid(self.logger.experiment, to_viz, 'projected', self.current_epoch)

        to_viz = p_1.detach()[0]
        to_viz = torch.argmax(to_viz, dim=0)
        save_grid(self.logger.experiment, to_viz, 'predicted', self.current_epoch)

        z_1, z_2, = z_1.detach(), z_2.detach()

        # p_1 = self.softmax(p_1)
        # p_2 = self.softmax(p_2)

        # z_1 = self.logmax(z_1)
        # z_2 = self.logmax(z_2)

        #loss = (self.loss(p_1[mask], z_2[mask]) + self.loss(p_2[mask], z_1[mask])) *0.5
        loss = (self.loss(p_1, z_2) + self.loss(p_2, z_1)) *0.5

        self.log('train_loss', loss)

        return loss/8

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=5e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 10)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "train_loss"}

    def depatch(self, x):
        x = rearrange(x, 'b (h w) c -> b c h w', h=self.num_patches, w=self.num_patches)
        #x = f.interpolate(x, scale_factor=2)
        return x

    def forward(self, x):
        x = self.embed(x)
        x = self.t_enc(x)
        x = self.map(x)
        x = self.depatch(x)
        x = self.proj(x)
        x = self.pred(x)
        x = torch.argmax(x, dim=1)
        return x



class MaskedSiam(pl.LightningModule):
    def __init__(self, num_channels, batch_size = 512, proj_d = 1024):
        super().__init__()
        self.batch_size = batch_size

        encoder_layer = torch.nn.TransformerEncoderLayer(d_model=num_channels, nhead=num_channels//6, dim_feedforward=1024, batch_first=True)
        self.t_enc = torch.nn.TransformerEncoder(encoder_layer, num_layers=9)
        self.proj = nn.Sequential(nn.Linear(num_channels, proj_d),
                                    nn.BatchNorm1d(proj_d),
                                    nn.ReLU(),
                                    nn.Linear(proj_d, proj_d),
                                    nn.BatchNorm1d(proj_d),
                                    nn.ReLU(),
                                    nn.Linear(proj_d, proj_d),
                                    nn.BatchNorm1d(proj_d)
                                    )

        self.pred = nn.Sequential(nn.Linear(proj_d, proj_d//4),
                                    nn.BatchNorm1d(proj_d//4),
                                    nn.ReLU(),
                                    nn.Linear(proj_d//4, proj_d)
                                    )

        self.loss = networks.SiamLoss()

    def training_step(self, x):
        inp, inp_aug, mask = x['base'].squeeze(), x['augment'].squeeze(), x['mask'].squeeze()
        z_1 = self.t_enc(inp)
        z_2 = self.t_enc(inp_aug)

        z_1 = self.proj(z_1)
        z_2 = self.proj(z_2)

        p_1 = self.pred(z_1)
        p_2 = self.pred(z_2)

        z_1, z_2, = z_1.detach(), z_2.detach()

        loss = (self.loss(p_1, z_2) + self.loss(p_2, z_1)) *0.5

        self.log('train_loss', loss)

        return loss/self.batch_size

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=5e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 10)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "train_loss"}

    def forward(self, x, mean, std):
        #Assumes we have numpy array (h w c)


        test_lin = torch.nn.Linear(1024, 60, bias=False)
        
        batched = rearrange(x, '(b1 h) (b2 w) c -> (b1 b2) (h w) c', b1=10, b2=10, h=100, w=100)
        batched = (batched - std)/mean
        holder = np.empty(batched.shape[0:2], dtype=np.int64)
        for ix, img in enumerate(batched):
            #orig_shape = img.shape
            #img = rearrange(img, 'h w c -> (h w) c')
            
            img = ma.masked_invalid(img)
            mask = img.mask.mean(axis=1).astype(np.bool)
            if mask.sum() < 100 * 100:
                img = ma.compress_rows(img)
                img = torch.from_numpy(img)

                img = self.t_enc(img)
                img = self.proj(img)
                img = self.pred(img)

                #img = f.log_softmax(img, dim=1)
                #img = f.sigmoid(img)
                #img = test_lin(img)

                img = torch.argmax(img, dim=1)

                img = img.detach().numpy()

                out = np.empty(mask.shape, dtype=np.int64)
                np.place(out, mask, -1)
                np.place(out, ~mask, img)
                img = out
                #img = rearrange(out, '(h w) -> h w', h=orig_shape[0], w=orig_shape[1])
            else: 
                img = np.ones((100*100), dtype=np.int64) * -1
            holder[ix] = img

        img = rearrange(holder, '(b1 b2) (h w) -> (b1 h) (b2 w)', b1=10, b2=10, h=100, w=100)


        return img

class MixedModel(pl.LightningModule):
    def __init__(self, num_channels, **kwargs):
        super().__init__()

        #Transformer
        self.patch_embed = networks.PatchEmbedding(in_channels=num_channels, patch_size=5, emb_size=25*num_channels, img_size=25)
        self.t_enc = networks.T_Enc(num_channels)
       
        #CNN
        self.c_enc = networks.C_Enc(num_channels)
        self.c_proj = networks.C_Proj(num_channels)
        #self.c_dec = networks.C_Dec(num_channels)

        #Voter?


        #Scaling and unpatching
        self.scale_up = nn.Upsample(scale_factor=5, mode='bilinear')
        self.depatch = networks.DePatch(num_channels=25*num_channels)

        #Loss 
        self.sim_loss = networks.SiamLoss()
        self.ce_loss = nn.CrossEntropyLoss()
   
    def training_step(self, x, idx, optimizer_idx):
        

        if optimizer_idx == 0:
            inp, inp_aug = x["base"].squeeze(), x["rand"].squeeze() 

            p_1 = self.t_enc(inp)
            p_2 = self.c_enc(inp_aug)
            p_2 = self.c_proj(p_2).detach()
            viz_p = self.depatch(p_1)

            self.log_images(viz_p)

            loss = self.sim_loss(p_1, p_2).mean() #+ self.loss(p_2, z_1).mean()) *0.5

            self.log('former_loss', loss)

            return loss/256

        if optimizer_idx == 1:
            inp, inp_aug = x["base"].squeeze(), x["rand"].squeeze() 

            p_1 = self.c_enc(inp)
            p_1 = self.c_proj(p_1)
            p_2 = self.t_enc(inp_aug).detach()
            # viz_p = self.depatch(p_1)

            # self.log_images(viz_p)

            loss = self.sim_loss(p_1, p_2).mean() #+ self.loss(p_2, z_1).mean()) *0.5

            self.log('conv_loss', loss)

            return loss/256

        
        # if optimizer_idx == 2:
        #     #MAKE THIS LEARNABLE??
        #     if torch.rand(1) < 0.10:
        #         inp = x['full']
        #         p_1 = self.c_enc(inp)
        #         p_1 = self.c_dec(p_1)
        #         p_1 = f.softmax(p_1, dim=1)
        #         p_1 = f.interpolate(p_1, size=(1000,1000)).squeeze()
        #         inp = inp.squeeze()
        #         batched = rearrange(inp, "c (b1 h) (b2 w) -> (b1 b2) c h w", h=25, w=25, b1=40, b2=40)
        #         p_2 = self.t_enc(batched)
        #         p_2 = self.depatch(p_2)

        #         p_2 = rearrange(p_2, "(b1 b2) c h w -> c (b1 h) (b2 w)", h=25, w=25, b1=40, b2=40)
        #         # p_2, z_2 = self.t_enc(inp_aug)

        #         # loss = self.sim_loss(p_1, z_2).mean() #+ self.loss(p_2, z_1).mean()) *0.5

        #         loss = self.ce_loss(p_1, p_2)
        #         self.log('conv_loss', loss)

        #         return loss

    def log_images(self, pred):
        tb = self.logger.experiment
        sample_pred = pred[:6]
        sample_pred = get_classifications(sample_pred)
        save_grid(tb, sample_pred, 'predicted', self.current_epoch)

    def configure_optimizers(self):
        optimizer_t = torch.optim.Adam(self.t_enc.parameters(), lr=5e-5)
        scheduler_t = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_t, 'min', patience=1, verbose=True)

        optimizer_c = torch.optim.Adam([{'params': self.c_enc.parameters(), 'params': self.c_proj.parameters()}], lr=5e-5)
        scheduler_c = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_c, 'min', patience=1, verbose=True)

        #optimizer_c_dec = torch.optim.Adam([{'params': self.c_enc.parameters(), 'params': self.c_dec.parameters()}], lr=5e-5)
        #return {"optimizers": [optimizer_t, optimizer_c], "lr_scheduler": [scheduler_t, scheduler_c], "monitor": "train_loss"}
        return [optimizer_t, optimizer_c], []
        
    def forward(self, x):
        #x = self.patch_embed(x)
        x  = self.t_enc(x)
        #x = self.proj_1(x)
        #x = self.pred_1(x)
        x = self.depatch(x)

        return x


class BYOLTransformer(pl.LightningModule):
    def __init__(self, num_channels, **kwargs):
        super().__init__()
        self.patch_embed = networks.PatchEmbedding(in_channels=num_channels, patch_size=5, emb_size=25*num_channels, img_size=25)
        encoder_layer = torch.nn.TransformerEncoderLayer(d_model=25*num_channels, nhead=15, dim_feedforward=1024, batch_first=True)
        self.online_enc = torch.nn.TransformerEncoder(encoder_layer, num_layers=4)
        #self.target_enc = torch.nn.TransformerEncoder(encoder_layer, num_layers=2)
        # self.proj = networks.TProjector()
        # self.pred = networks.TPredictor()
        self.proj_1 = networks.SegLinear(num_channels=25 * num_channels, b1=25, b2=25)
        #self.proj_2 = networks.SegLinear(num_channels=676, b1=1024, b2=676)

        self.pred_1 = networks.SegDecoder(num_channels = 25 * num_channels, patches=25)
        #self.pred_2 = networks.SegDecoder(num_channels=676, num_classes=676, drop_class=False, patches=676)
        #self.online_proj = networks.BYOLLinear(270)
        #self.loss = nn.MSELoss()
        #self.loss_2 = nn.KLDivLoss(reduction='batchmean')
        # self.softmax = torch.nn.Softmax(dim=1)
        # self.logmax = torch.nn.LogSoftmax(dim=1)
        self.scale_up = nn.Upsample(scale_factor=5, mode='bilinear')
        self.depatch = networks.DePatch(num_channels=25*num_channels)
        self.loss = networks.SiamLoss()
        #self.online_pred = networks.BYOLLinear(270)
        #self.target_proj = networks.BYOLLinear(270)

    
    def training_step(self, x, idx):
        #viz, inp, inp_aug = x["viz"].squeeze(), x["base"].squeeze(), x["rand"].squeeze()
        inp, inp_aug = x["base"].squeeze(), x["rand"].squeeze()

        x_1 = self.patch_embed(inp)
        #x_d = self.decoder(x_1)
        x_2 = self.patch_embed(inp_aug)

        x_1 = self.online_enc(x_1)
        z_1 = self.proj_1(x_1)
        p_1 = self.pred_1(z_1)

        x_2 = self.online_enc(x_2)
        z_2 = self.proj_1(x_2)
        p_2 = self.pred_1(z_2)

        viz_p = self.depatch(p_1)

        self.log_images(viz_p)
        
        z1_stop = z_1.detach()
        z2_stop = z_2.detach()

        # e1 = torch.matmul(x_1, z_1.moveaxis(2, 1)).flatten(start_dim=1)
        # e2 = torch.matmul(x_2, z_2.moveaxis(2, 1)).flatten(start_dim=1)


        # v1 = self.proj_2(e1)
        # u1 = self.pred_2(v1)

        # v2 = self.proj_2(e2)

        #self.log_images(viz,z1,p1, inp, inp_aug)
        pix_loss = (self.loss(p_1, z2_stop).mean() + self.loss(p_2, z1_stop).mean()) *0.5
        #pix_loss = (-(self.softmax(p_1).mean() * self.logmax(z2_stop).mean())-(self.softmax(p_2).mean() * self.logmax(z1_stop).mean())) * 0.5
        #region_loss = self.loss(u1, v2)/15
        #rand_loss = self.loss(p_1, torch.randn_like(p_1)) *.005
        loss = (pix_loss)/256  #* 0.5
        #k_l_loss = self.loss_2(u1.log(), v2)
        # z_2 = z_2.detach()
        # z_1 = z_1.detach()

        # loss = (self.loss(p_1, z_2) + self.loss(p_2, z_1)) * 0.5
        self.log('train_loss', loss)
        return loss

    def log_images(self, pred):
        tb = self.logger.experiment
         

        sample_pred = pred[:6]
        #sample_pred = self.scale_up(sample_pred)
        sample_pred = get_classifications(sample_pred)
        save_grid(tb, sample_pred, 'predicted', self.current_epoch)



    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=5e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=1, verbose=True)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "train_loss"}
        
    def forward(self, x):
        x =self.patch_embed(x)
        x = self.online_enc(x)
        x = self.proj_1(x)
        x = self.pred_1(x)
        x = self.depatch(x)
        #x = self.scale_up(x)

        return x
    




class DenseSimSiam(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        #self.encoder = net_gen.ResnetEncoder(kwargs["num_channels"], 512, use_dropout=True)
        self.encoder = net_gen.ResnetGenerator(kwargs["num_channels"], 20, n_blocks =9)
        self.projector_1 = networks.DenseProjector(num_channels=20)
        self.predictor_1 = networks.DensePredictor(num_classes=20)
        #self.projector_2 = networks.DenseProjectorMLP()
        self.projector_2 = networks.DenseProjector(num_channels=20)
        #self.predictor_2 = networks.DensePredictorMLP()
        self.predictor_2 = networks.DensePredictor(num_classes=20)
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

        e1 = torch.matmul(x1, z1) #.flatten(start_dim=2)
        e2 = torch.matmul(x2, z2) #.flatten(start_dim=2)


        v1 = self.projector_2(e1)
        u1 = self.predictor_2(v1)

        v2 = self.projector_2(e2)

        self.log_images(viz,z1,p1, inp, inp_aug)

        pix_loss = (-(self.softmax(p1).mean() * self.logmax(z2_stop).mean())-(self.softmax(p2).mean() * self.logmax(z1_stop).mean())) * 0.5
        region_loss = self.loss(u1, v2) *.25
        loss = (pix_loss + region_loss) #* 0.5
        self.log('train_loss', loss)
        return loss

    def log_images(self, viz, proj, pred, x, x1):
        tb = self.logger.experiment
        sample_imgs = viz[:6]
        save_grid(tb, sample_imgs, 'rgb', self.current_epoch)
           
        sample_proj = proj[:6]
        sample_proj = get_classifications(sample_proj)
        save_grid(tb, sample_proj, 'projected', self.current_epoch)

        sample_pred = pred[:6]
        sample_pred = get_classifications(sample_pred)
        save_grid(tb, sample_proj, 'predicted', self.current_epoch)

        sample_x = x[:6, 3:6]
        save_grid(tb, sample_x, 'sample_x', self.current_epoch)

        sample_x1 = x1[:6, 3:6]
        save_grid(tb, sample_x1, 'sample_x1', self.current_epoch)


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=5e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=1, verbose=True)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "train_loss"}
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.projector_1(x)
        return x
    


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

def save_grid(tb, inp, name, epoch):  
    #img_grid = rearrange(inp, 'b h w -> h (b w)')
    tb.add_image(name, inp, epoch, dataformats='HW')

def get_classifications(x):
    #norms = torch.nn.functional.softmax(x, dim=1)
    masks = x.argmax(1)

    # masks = torch.unsqueeze(masks, 1) * 255.0/59
    # masks = torch.cat((masks, masks, masks), dim=1)
    return masks