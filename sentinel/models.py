import torch.nn as nn
import torchvision.transforms as tt
import transforms as tr
import torch
from einops.layers.torch import Rearrange
from einops import rearrange, reduce
import torch.nn.functional as f
import pytorch_lightning as pl
import torchvision.transforms.functional as TF


class SwaVModelUnified(pl.LightningModule):
    def __init__(self, 
                class_key,
                lr, 
                class_weights, 
                num_intermediate_classes,
                pre_training,
                predict_mode='up', 
                augment_refine = 'false',
                scheduler=True,
                positions=False,
                emb_size=256,
                augment_bright=False,
                patch_size=4):
        super().__init__()
        self.save_hyperparameters()
        self.num_channels = 12
        self.num_output_classes = len(class_key.keys())
        self.missing_data = nn.Parameter(torch.randn((1)))
        self.lr = lr
        self.augment_refine = augment_refine
        self.augment_bright = augment_bright
        if self.augment_bright:
            self.ab = tr.BrightnessAugment(p=1)
        self.mode = predict_mode
        self.scheduler = scheduler

        self.swav = SWaVUnifiedPerPatch(self.num_channels, num_intermediate_classes, n_head=8, n_layers=4, patch_size=patch_size, positions=positions, emb_size=emb_size)
        if self.mode == 'default':
            self.predict =  nn.Sequential(nn.Linear(num_intermediate_classes, num_intermediate_classes*2),
                                         nn.BatchNorm1d(num_intermediate_classes*2),
                                         nn.ReLU(),
                                         nn.Linear(num_intermediate_classes*2, self.num_output_classes))
        else:
            self.predict = UpsamplePredictor(num_intermediate_classes, self.num_output_classes)
        if class_weights is not None:
            self.loss = nn.CrossEntropyLoss(weight=torch.tensor(class_weights), label_smoothing=0.2, ignore_index=-1)
        else:
            self.loss = nn.CrossEntropyLoss(label_smoothing=0.2, ignore_index=-1)

        self.class_key = {value:key for key, value in class_key.items()}
        self.results = self.make_results_dict(self.class_key)
        self.pre_training = pre_training
        self.ra = Rearrange('b c h w -> b (h w) c')


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


    def pre_training_step(self, inp):
        inp = inp['scenes']

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

        inp = inp['scenes']

        missing_mask = inp != inp
        missing_mask = reduce(missing_mask, 'b c h w -> b h w', 'max')


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
        

        inp[inp != inp] = self.missing_data
            
        inp = self.swav.forward(inp)

        inp = rearrange(inp, 'b (h w) f -> b f h w', h=16, w=16)
        inp = self.predict(inp)


        inp = torch.softmax(inp, dim=1)
        #targets = targets.type(torch.LongTensor)

        #targets=torch.argmax(targets, dim=1)
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
        x = x['scenes']
        out = self.swav.forward(x)
        out = rearrange(out, 'b s f -> (b s) f')
        out = self.predict(out)
        return out



class UpsamplePredictor(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpsamplePredictor, self).__init__()
        self.up1 = nn.Sequential(nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2),
                                nn.InstanceNorm2d(in_channels),
                                nn.ReLU())
        self.up2 = nn.Sequential(nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2),
                                nn.InstanceNorm2d(in_channels),
                                nn.ReLU())
        self.max = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, in_channels//2, kernel_size=3, padding=1),
                                nn.InstanceNorm2d(in_channels//2),
                                nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels//2, in_channels//4, kernel_size=3, padding=1),
                                nn.InstanceNorm2d(in_channels//4),
                                nn.ReLU())
        self.up3 = nn.Upsample(scale_factor =2, mode='bilinear')
        self.classify = nn.Conv2d(in_channels//4, out_channels, kernel_size=1)
    
    def forward(self, x):
        x = self.up1(x)
        x = self.up2(x)
        x = self.max(x)
        x = self.conv1(x)
        x= self.conv2(x)
        x = self.up3(x)
       # x = self.up3(x)
        x = self.classify(x)
        return x


class SWaVUnifiedPerPatch(nn.Module):
    def __init__(self, 
                in_channels,
                num_classes, 
                n_head = 8,
                n_layers = 8,
                emb_size=256, 
                temp=0.1, 
                epsilon=0.05,  
                sinkhorn_iters=3,
                patch_size=4,
                positions=False):
        super(SWaVUnifiedPerPatch, self).__init__()

        self.missing = nn.Parameter(torch.randn((1)))
        
        self.transforms_main = tt.Compose([
                                        tr.Blit(self.missing, p=0.5),
                                        tr.Block(self.missing, p=0.5),
                                        tr.PatchBlock(self.missing, p=0.5)
                                       ])


        self.patches = Rearrange('b c (h s1) (w s2) -> b (h w) (s1 s2 c)', s1=patch_size, s2=patch_size)
        self.use_positions = positions

        self.positions = nn.Parameter(torch.randn((64, emb_size)))


        self.embed = nn.Linear(in_channels * patch_size**2, emb_size)

        encoder_layer = torch.nn.TransformerEncoderLayer(d_model=emb_size, nhead=n_head, dim_feedforward=emb_size*2, batch_first=True)
        self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.projector = nn.Sequential(nn.Linear(emb_size, emb_size),
                                        nn.ReLU(),
                                        nn.Linear(emb_size, emb_size),
                                        nn.ReLU(),
                                        nn.Linear(emb_size, emb_size))

        self.prototypes = nn.Linear(emb_size, num_classes, bias=False)

        self.softmax = nn.LogSoftmax(dim=1)
        self.temp = temp
        self.epsilon = epsilon
        self.niters = sinkhorn_iters
        self.ra = Rearrange('b s f -> (b s) f')

    @torch.no_grad()
    def norm_prototypes(self):
        w = self.prototypes.weight.data.clone()
        w = f.normalize(w, dim=1, p=2)
        self.prototypes.weight.copy_(w)

    @torch.no_grad()
    def sinkhorn(self, scores):
        Q = torch.exp(scores/self.epsilon).t()
        B = Q.shape[1] 
        K = Q.shape[0]

        sum_Q = torch.sum(Q)
        Q /= sum_Q

        for it in range(self.niters):
            # normalize each row: total weight per prototype must be 1/K
            sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
            Q /= sum_of_rows
            Q /= K

            # normalize each column: total weight per sample must be 1/B
            Q /= torch.sum(Q, dim=0, keepdim=True)
            Q /= B

        Q *= B # the colomns must sum to 1 so that Q is an assignment
        return Q.t()


    # def forward_train(self, x, chm, azm):
    def forward_train(self, x):

        b = x.shape[0]

        x = self.patches(x)

        x_s = self.transforms_main(x)

        x = self.embed(x)

        
        x_s = self.embed(x_s)
        if self.use_positions:
            x = x + self.positions
            x_s = x_s + self.positions


        inp = torch.cat((x, x_s))
        inp = self.encoder(inp)
        inp = self.projector(inp)
        inp = nn.functional.normalize(inp, dim=1, p=2)

        scores = self.prototypes(inp)


        scores_t = scores[:b]
        scores_s = scores[b:]

        t = scores_t.detach()
        s = scores_s.detach()

        t = self.ra(t)
        s = self.ra(s)

        scores_t = self.ra(scores_t)
        scores_s = self.ra(scores_s)

        b = scores_t.shape[0]


        q_t = self.sinkhorn(t)
        q_s = self.sinkhorn(s)

        p_t = self.softmax(scores_t/self.temp)
        p_s = self.softmax(scores_s/self.temp)

        loss = -0.5 * torch.mean(q_t * p_s + q_s * p_t)

        return loss
    
    def forward(self, inp):
        inp[inp != inp] = self.missing

        inp = self.patches(inp)
       
        inp = self.embed(inp)
        if self.use_positions:
            inp = inp + self.positions

        inp = self.encoder(inp)
        inp = self.projector(inp)
        inp = nn.functional.normalize(inp, dim=1, p=2)

        scores = self.prototypes(inp)
        return scores