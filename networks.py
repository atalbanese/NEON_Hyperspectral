from requests import patch
import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as f
from transformers import PerceiverForImageClassificationConvProcessing
import net_gen
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
import numpy as np
import torchvision.transforms as tt
import transforms as tr

class UpsamplePredictor(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpsamplePredictor, self).__init__()
        self.up1 = nn.Sequential(nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2),
                                nn.BatchNorm2d(in_channels),
                                nn.ReLU())
        self.up2 = nn.Sequential(nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2),
                                nn.BatchNorm2d(in_channels),
                                nn.ReLU())
        self.max = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, in_channels//2, kernel_size=3, padding=1),
                                nn.BatchNorm2d(in_channels//2),
                                nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels//2, in_channels//4, kernel_size=3, padding=1),
                                nn.BatchNorm2d(in_channels//4),
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
        x = self.classify(x)
        return x


class SWaVUnifiedPerPixelPatch(nn.Module):
    def __init__(self, 
                in_channels,
                num_classes, 
                n_head = 8,
                n_layers = 8,
                emb_size=128, 
                temp=0.1, 
                epsilon=0.05,  
                sinkhorn_iters=3):
        super(SWaVUnifiedPerPixelPatch, self).__init__()
        
        self.transforms_main = tt.Compose([
                                        tr.Blit(p=0.5),
                                        tr.Block(p=0.5)
                                       ])


        self.embed = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, emb_size, kernel_size=(3,3), stride=1, padding=0),
            Rearrange('b c h w -> b (h w) c')
        )

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
        x_s = self.transforms_main(x)

        x = self.embed(x)
        
        x_s = self.embed(x_s)


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

        return loss/b
    
    def forward(self, inp):
       
        inp = self.embed(inp)

        inp = self.encoder(inp)
        inp = self.projector(inp)
        inp = nn.functional.normalize(inp, dim=1, p=2)

        scores = self.prototypes(inp)
        return scores



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
        
        self.transforms_main = tt.Compose([
                                        #tr.Blit(p=0.5),
                                        tr.PatchBlock(p=1)
                                       ])

        self.patches = Rearrange('b c (h s1) (w s2) -> b (h w) (s1 s2 c)', s1=patch_size, s2=patch_size)
        self.use_positions = positions

        self.positions = nn.Parameter(torch.randn((25, emb_size)))


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

        inp = self.patches(inp)
       
        inp = self.embed(inp)
        if self.use_positions:
            inp = inp + self.positions

        inp = self.encoder(inp)
        inp = self.projector(inp)
        inp = nn.functional.normalize(inp, dim=1, p=2)

        scores = self.prototypes(inp)
        return scores


class SWaVUnified(nn.Module):
    def __init__(self, 
                in_channels,
                num_classes, 
                n_head = 8,
                n_layers = 8,
                emb_size=128, 
                temp=0.1, 
                epsilon=0.05,  
                sinkhorn_iters=3,
                positions=False):
        super(SWaVUnified, self).__init__()
        
        self.transforms_main = tt.Compose([
                                        tr.Blit(p=0.5),
                                        tr.Block(p=0.5)
                                       ])


        self.embed = nn.Linear(in_channels, emb_size)
        self.use_positions = positions

        if positions:
            self.positions = nn.Parameter(torch.randn((9, emb_size)))

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
        x_s = self.transforms_main(x)

        x = self.embed(x)
        x_s = self.embed(x_s)
        

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

        return loss/b


    def forward(self, inp):
       
        inp = self.embed(inp)
        if self.use_positions:
            inp = inp + self.positions


        inp = self.encoder(inp)
        inp = self.projector(inp)
        inp = nn.functional.normalize(inp, dim=1, p=2)

        scores = self.prototypes(inp)
        return scores

class SWaVSuperPixel(nn.Module):
    def __init__(self, 
                in_channels=31, 
                emb_size=128, 
                temp=0.1, 
                epsilon=0.05, 
                patch_size = 4, 
                sinkhorn_iters=3, 
                num_classes=12, 
                azm=False, 
                chm=False, 
                queue_chunks=5, 
                azm_concat=False, 
                chm_concat=False,
                positions=False,
                batch_size=2048):
        super(SWaVSuperPixel, self).__init__()
        self.populate_queue = False
        self.use_queue = False
        #TODO: Fix the queue
        self.queue_base = torch.zeros((queue_chunks)*batch_size*patch_size**2, emb_size).cuda()
        self.queue_mod = torch.zeros((queue_chunks)*batch_size*patch_size**2, emb_size).cuda()
        self.chm = chm
        self.azm = azm
        self.azm_concat = azm_concat
        self.chm_concat = chm_concat
        self.patch_size = patch_size

        # self.add_channel = 0
        # if self.azm_concat:
        #     self.add_channel += 1
        # if self.chm_concat:
        #     self.add_channel +=1

        self.queue_chunks = queue_chunks

        self.use_positions = positions
        if positions:
            self.positions = nn.Parameter(torch.randn((self.patch_size**2, emb_size)))


        self.transforms_main = tt.Compose([
                                        tr.Blit(p=0.5),
                                        tr.Block(p=0.5)
                                       ])

        #Todo: add positional embedding
        self.embed = nn.Linear(in_channels, emb_size)


        # self.azm_embed = nn.Sequential(
        #                 Rearrange('b c h w -> b (h w) c'),
        #                 nn.Linear(1, emb_size)
        # )

        # self.chm_embed = nn.Sequential(
        #                 Rearrange('b c h w -> b (h w) c'),
        #                 nn.Linear(1, emb_size)
        # )

        # self.transforms_embed = tt.Compose([Rearrange('b c h w -> b (h w) c'),
        #                                 tr.Blit(),
        #                                 Rearrange('b (h w) c -> b c h w', h=self.patch_size, w=self.patch_size)])

     
        encoder_layer = torch.nn.TransformerEncoderLayer(d_model=emb_size, nhead=8, dim_feedforward=emb_size*2, batch_first=True)
        self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=8)

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

        # if self.chm_concat:
        #     x = torch.cat((x, chm), dim=1)
        # if self.azm_concat:
        #     x= torch.cat((x, azm), dim=1)
        

        x_s = self.transforms_main(x)

        x = self.embed(x)
        
        x_s = self.embed(x_s)
        # if self.use_positions:
        #     x += self.positions
        #     x_s += self.positions

        # if self.chm and not self.chm_concat:
        #     chm_s = self.transforms_embed(chm)
        #     chm = self.chm_embed(chm)
        #     chm_s = self.chm_embed(chm_s)
        #     x += chm
        #     x_s += chm_s

        # if self.azm and not self.azm_concat:
        #     azm_s = self.transforms_embed(azm)
        #     azm = self.azm_embed(azm)
        #     azm_s = self.azm_embed(azm_s)
        #     x += azm
        #     x_s += azm_s

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

        o_b = b
        b *= self.patch_size**2

        if self.use_queue:

            t = torch.cat((
                torch.mm(
                    self.queue_base,
                    self.prototypes.weight.t()
                ),
                t
            ))

            s = torch.cat((
                torch.mm(
                    self.queue_mod,
                    self.prototypes.weight.t()
                ),
                s
            ))

            q_t = self.sinkhorn(t)[-b:]
            q_s = self.sinkhorn(s)[-b:]
        else:
            q_t = self.sinkhorn(t)
            q_s = self.sinkhorn(s)

        p_t = self.softmax(scores_t/self.temp)
        p_s = self.softmax(scores_s/self.temp)

        loss = -0.5 * torch.mean(q_t * p_s + q_s * p_t)

        #b = q_t.shape[0]

        if self.populate_queue:
            with torch.no_grad():
                base = inp[:o_b].detach()
                base = self.ra(base)
                self.queue_base[b:] = self.queue_base[:-b].clone()
                self.queue_base[:b] = base

                mod = inp[o_b:].detach()
                mod = self.ra(mod)
                self.queue_mod[b:] = self.queue_mod[:-b].clone()
                self.queue_mod[:b] = mod


        #Try with and without /b
        return loss/b


    #TODO: Fix This
    def forward(self, inp):
       
        inp = self.embed(inp)

        inp = self.encoder(inp)
        inp = self.projector(inp)
        inp = nn.functional.normalize(inp, dim=1, p=2)

        scores = self.prototypes(inp)
        return scores

class SWaVStruct(nn.Module):
    def __init__(self, img_size=40, patch_size=2, in_channels=10, emb_size=256, temp=0.1, epsilon=0.05, sinkhorn_iters=3, num_classes=12, azm=True, chm=True, same_embed=False, concat=False, queue_chunks=1, azm_concat=False, chm_concat=False, aug_brightness=False):
        super(SWaVStruct, self).__init__()
        self.populate_queue = False
        self.use_queue = False
        self.num_patches = (img_size*img_size)//(patch_size*patch_size)
        self.queue_base = torch.zeros((queue_chunks+1)*256, self.num_patches, emb_size).cuda()
        self.queue_mod = torch.zeros((queue_chunks+1)*256, self.num_patches, emb_size).cuda()
        self.chm = chm
        self.azm = azm
        self.azm_concat = azm_concat
        self.chm_concat = chm_concat

        self.same_embed = same_embed
        # self.concat = concat
        # self.add_channel = 2 if self.concat else 0

        self.add_channel = 0
        if self.azm_concat:
            self.add_channel += 1
        if self.chm_concat:
            self.add_channel +=1

        self.queue_chunks = queue_chunks

        if aug_brightness:
            self.aug_brightness = tr.BrightnessAugment()
        else:
            self.aug_brightness = None


        self.transforms_main = tt.Compose([Rearrange('b c h w -> b (h w) c'),
                                        tr.Blit(),
                                        tr.Block(),
                                        Rearrange('b (h w) c -> b c h w', h=img_size, w=img_size)])
        

        self.transforms_embed = tt.Compose([Rearrange('b c h w -> b (h w) c'),
                                        tr.Blit(),
                                        Rearrange('b (h w) c -> b c h w', h=img_size, w=img_size)])

        self.patch_embed = nn.Sequential(
            # break-down the image in s1 x s2 patches and flat them
            Rearrange('b c (h s1) (w s2) -> b (h w) (s1 s2 c)', s1=patch_size, s2=patch_size),
            nn.Linear(patch_size * patch_size * (in_channels+self.add_channel), emb_size)
        )
        if self.azm:
            self.azm_embed = nn.Sequential(
                # break-down the image in s1 x s2 patches and flat them
                Rearrange('b c (h s1) (w s2) -> b (h w) (s1 s2 c)', s1=patch_size, s2=patch_size),
                nn.Linear(patch_size * patch_size, emb_size)
            )
        
        if self.chm and not self.same_embed:
            self.chm_embed = nn.Sequential(
                # break-down the image in s1 x s2 patches and flat them
                Rearrange('b c (h s1) (w s2) -> b (h w) (s1 s2 c)', s1=patch_size, s2=patch_size),
                nn.Linear(patch_size * patch_size, emb_size)
            )


        encoder_layer = torch.nn.TransformerEncoderLayer(d_model=emb_size, nhead=4, dim_feedforward=emb_size*2, batch_first=True)
        self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=4)

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

    #TODO: freeze protos for first epoch
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


    def forward_train(self, x, chm, azm):

        b,_,_,_ = x.shape

        if self.aug_brightness is not None:
            x = self.aug_brightness(x)

        if self.chm_concat:
            x = torch.cat((x, chm), dim=1)
        if self.azm_concat:
            x= torch.cat((x, azm), dim=1)

        x_s = self.transforms_main(x)
        x = self.patch_embed(x)
        x_s = self.patch_embed(x_s)

        if self.chm and not self.chm_concat:
            chm_s = self.transforms_embed(chm)
            if not self.same_embed:
                chm = self.chm_embed(chm)
                chm_s = self.chm_embed(chm_s)
            else:
                chm = self.azm_embed(chm)
                chm_s = self.azm_embed(chm_s)
            x += chm
            x_s += chm_s

        if self.azm and not self.azm_concat:
            azm_s = self.transforms_embed(azm)
            azm = self.azm_embed(azm)
            azm_s = self.azm_embed(azm_s)
            x += azm
            x_s += azm_s


        inp = torch.cat((x, x_s))
        inp = self.encoder(inp)
        inp = self.projector(inp)
        inp = nn.functional.normalize(inp, dim=1, p=2)

        scores = self.prototypes(inp)
        #scores = scores.detach()

        if self.populate_queue:
            with torch.no_grad():
                self.queue_base[b:] = self.queue_base[:-b].clone()
                self.queue_base[:b] = inp[:b].detach()
                self.queue_mod[b:] = self.queue_mod[:-b].clone()
                self.queue_mod[:b] = inp[b:].detach()

        scores_t = scores[:b]
        scores_s = scores[b:]

        t = scores_t.detach()
        s = scores_s.detach()

        loss = 0

        for i in range(b):
            t_i, s_i = t[i], s[i]

            

            if self.use_queue:
                qb_items = torch.cat(([self.queue_base[i+j] for j in range(self.queue_chunks)]))
                qm_items = torch.cat(([self.queue_mod[i+j] for j in range(self.queue_chunks)]))

                t_i = torch.cat((
                    torch.mm(
                        qb_items,
                        self.prototypes.weight.t()
                    ),
                    t_i
                ))

                s_i = torch.cat((
                    torch.mm(
                        qm_items,
                        self.prototypes.weight.t()
                    ),
                    s_i
                ))

                q_t = self.sinkhorn(t_i)[-self.num_patches:]
                q_s = self.sinkhorn(s_i)[-self.num_patches:]
            else:
                q_t = self.sinkhorn(t_i)
                q_s = self.sinkhorn(s_i)

            p_t = self.softmax(scores_t[i]/self.temp)
            p_s = self.softmax(scores_s[i]/self.temp)

            loss += -0.5 * torch.mean(q_t * p_s + q_s * p_t)

        return loss/b

    def forward(self, inp, chm, azm):
        if self.chm_concat:
            inp = torch.cat((inp, chm), dim=1)
        if self.azm_concat:
            inp = torch.cat((inp, azm), dim=1)

        inp = self.patch_embed(inp)
        if self.chm and not self.chm_concat:
            if not self.same_embed:
                chm = self.chm_embed(chm)
            else:
                chm = self.azm_embed(chm)
            inp += chm
        if self.azm and not self.azm_concat:
            azm= self.azm_embed(azm)
            inp += azm
        inp = self.encoder(inp)
        inp = self.projector(inp)
        inp = nn.functional.normalize(inp, dim=1, p=2)

        scores = self.prototypes(inp)
        return scores

class SWaVResnet(nn.Module):
    def __init__(self, img_size=40, patch_size=2, in_channels=10, emb_size=256, temp=0.1, epsilon=0.05, sinkhorn_iters=3, num_classes=12):
        super(SWaVResnet, self).__init__()
        self.transforms_1 =  tt.Compose([tt.RandomHorizontalFlip(),
                                    tt.RandomVerticalFlip()])

        self.transforms_2 = tt.Compose([Rearrange('b c h w -> b (h w) c'),
                                        tr.Blit(),
                                        tr.Block(),
                                        Rearrange('b (h w) c -> b c h w', h=img_size, w=img_size)])


        self.encoder = net_gen.ResnetGenerator(in_channels, num_classes, n_blocks=3)

        self.projector = DenseProjector(num_channels=num_classes)

        self.prototypes = nn.Conv2d(num_classes, num_classes, kernel_size=1, bias=False)

        self.softmax = nn.LogSoftmax(dim=1)

        self.flatten = Rearrange('c h w -> c (h w)')
        self.temp = temp
        self.epsilon = epsilon
        self.niters = sinkhorn_iters

    #TODO: freeze protos for first epoch
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


    def forward_train(self, x):

        b,_,_,_ = x.shape

        x_t = self.transforms_1(x)
        x_s = self.transforms_2(x_t)

        inp = torch.cat((x_t, x_s))

        inp = self.encoder(inp)
        inp = self.projector(inp)
        inp = nn.functional.normalize(inp, dim=1, p=2)

        scores = self.prototypes(inp)

        scores_t = scores[:b]
        scores_s = scores[b:]

        loss = 0

        for i in range(b):
            t, s = scores_t[i].detach(), scores_s[i].detach()
            t = self.flatten(t)
            s = self.flatten(s)
            q_t = self.sinkhorn(t)
            q_s = self.sinkhorn(s)

            p_t = self.softmax(scores_t[i]/self.temp)
            p_s = self.softmax(scores_s[i]/self.temp)

            p_t = self.flatten(p_t)
            p_s = self.flatten(p_s)

            loss += -0.5 * torch.mean(q_t * p_s + q_s * p_t)

        return loss/b

    def forward(self, inp):
        inp = self.encoder(inp)
        inp = self.projector(inp)
        inp = nn.functional.normalize(inp, dim=1, p=2)

        scores = self.prototypes(inp)
        return scores

class SWaV(nn.Module):
    def __init__(self, img_size=40, patch_size=2, in_channels=10, emb_size=256, temp=0.1, epsilon=0.05, sinkhorn_iters=3, num_classes=12):
        super(SWaV, self).__init__()
        self.transforms_1 =  tt.Compose([tt.RandomHorizontalFlip(),
                                    tt.RandomVerticalFlip()])

        self.transforms_2 = tt.Compose([Rearrange('b c h w -> b (h w) c'),
                                        tr.Blit(),
                                        tr.Block(),
                                        Rearrange('b (h w) c -> b c h w', h=img_size, w=img_size)])

        self.patch_embed = nn.Sequential(
            # break-down the image in s1 x s2 patches and flat them
            Rearrange('b c (h s1) (w s2) -> b (h w) (s1 s2 c)', s1=patch_size, s2=patch_size),
            nn.Linear(patch_size * patch_size * in_channels, emb_size)
        )

        encoder_layer = torch.nn.TransformerEncoderLayer(d_model=emb_size, nhead=4, dim_feedforward=emb_size*2, batch_first=True)
        self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=4)

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

    #TODO: freeze protos for first epoch
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


    def forward_train(self, x):

        b,_,_,_ = x.shape

        x_t = self.transforms_1(x)
        x_s = self.transforms_2(x_t)

        inp = torch.cat((x_t, x_s))

        inp = self.patch_embed(inp)
        inp = self.encoder(inp)
        inp = self.projector(inp)
        inp = nn.functional.normalize(inp, dim=1, p=2)

        scores = self.prototypes(inp)

        scores_t = scores[:b]
        scores_s = scores[b:]

        loss = 0

        for i in range(b):
            t, s = scores_t[i].detach(), scores_s[i].detach()
            q_t = self.sinkhorn(t)
            q_s = self.sinkhorn(s)

            p_t = self.softmax(scores_t[i]/self.temp)
            p_s = self.softmax(scores_s[i]/self.temp)

            loss += -0.5 * torch.mean(q_t * p_s + q_s * p_t)

        return loss/b

    def forward(self, inp):
        inp = self.patch_embed(inp)
        inp = self.encoder(inp)
        inp = self.projector(inp)
        inp = nn.functional.normalize(inp, dim=1, p=2)

        scores = self.prototypes(inp)
        return scores
        


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
    def __init__(self, num_channels = 512):
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
        #x = f.interpolate(x, (25, 25))
        return x

#Predictor h - just guessing as to whats this is supposed to be like
class DensePredictor(nn.Module):
    def __init__(self, num_classes =10):
        super(DensePredictor, self).__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(num_classes, num_classes, kernel_size=1),
                                    nn.ReLU()
                                    )
        self.layer2 = nn.Conv2d(num_classes, num_classes, kernel_size=1, bias=False)
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)

        return x

class DenseProjectorMLP(nn.Module):
    def __init__(self, num_channels=512):
        super(DenseProjectorMLP, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(num_channels, num_channels//2),
                                    nn.GroupNorm(16, 256),
                                    nn.ReLU())
        self.layer2 = nn.Sequential(nn.Linear(num_channels//2, num_channels//4),
                                    nn.GroupNorm(16, 256),
                                    nn.ReLU())
        self.layer3 = nn.Sequential(nn.Linear(num_channels//4, num_channels//4),
                                    nn.GroupNorm(16, 256),
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



class ResnetPatchEmbed(nn.Module):
    def __init__(self, in_channels: int = 30, patch_size: int = 5, emb_size: int = 750, img_size=25):
        super(ResnetPatchEmbed, self).__init__()
        self.layer1 = net_gen.ResnetGenerator(in_channels, emb_size//(patch_size**2), n_blocks=3)
        self.rearrange = Rearrange('b c (h s1) (w s2) -> b (h w) (s1 s2 c)', s1=patch_size, s2=patch_size)

        self.cls_token = nn.Parameter(torch.randn(1,1, emb_size))
        self.positions = nn.Parameter(torch.randn((img_size // patch_size) **2 + 1, emb_size))

    def forward(self, x):
        b, _, _, _ = x.shape
        x = self.layer1(x)
        x= self.rearrange(x)
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)
        # prepend the cls token to the input
        x = torch.cat([cls_tokens, x], dim=1)
        # add position embedding
        x += self.positions
        return x


class LinearPatchEmbed(nn.Module):
    def __init__(self, in_channels: int = 30, patch_size: int = 5, emb_size: int = 750, img_size=25):
        super(LinearPatchEmbed, self).__init__()
        self.patch_size = patch_size
        self.projection = nn.Sequential(
            # break-down the image in s1 x s2 patches and flat them
            Rearrange('b c (h s1) (w s2) -> b (h w) (s1 s2 c)', s1=patch_size, s2=patch_size),
            nn.Linear(patch_size * patch_size * in_channels, emb_size)
        )
        self.cls_token = nn.Parameter(torch.randn(1,1, emb_size))
        #self.positions = nn.Parameter(torch.randn((img_size // patch_size) **2 + 1, emb_size))

                
    def forward(self, x):
        b, _, _, _ = x.shape
        x = self.projection(x)
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)
        # prepend the cls token to the input
        x = torch.cat([cls_tokens, x], dim=1)
        # add position embedding
        #x += self.positions
        return x


class RCU(nn.Module):
    def __init__(self, num_channels=256):
        super(RCU, self).__init__()
        self.layer1 = nn.Sequential(nn.ReLU(),
                                    nn.BatchNorm2d(num_channels),
                                    nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(),
                                    nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1))
    def forward(self, x):
        z = self.layer1(x)
        return x+z

class Fusion(nn.Module):
    def __init__(self, num_channels=256):
        super(Fusion, self).__init__()
        self.RCU1 = RCU(num_channels=num_channels)
        self.RCU2 = RCU(num_channels=num_channels)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear')
        #self.pool = ChainedPool(num_channels=num_channels)
        self.project = RCU(num_channels=num_channels)

    def forward(self, x, z=None):
        x = self.RCU1(x)
        if z is not None:
            x = x+z
        x = self.RCU2(x)
        x = self.up(x)
        #x = self.pool(x)
        x = self.project(x)
        return x

class ChainedPool(nn.Module):
    def __init__(self, num_channels=256):
        super(ChainedPool, self).__init__()
        self.layer1 = nn.ReLU()
        self.pool1 = nn.Sequential(nn.MaxPool2d(kernel_size=5, stride=1, padding=2),
                                    nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1))
        self.pool2 = nn.Sequential(nn.MaxPool2d(kernel_size=5, stride=1, padding=2),
                                    nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1))
        self.pool3 = nn.Sequential(nn.MaxPool2d(kernel_size=5, stride=1, padding=2),
                                    nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1))


    def forward(self, x):
        x = self.layer1(x)
        z = self.pool1(x)
        x = x + z
        z = self.pool2(z)
        x = x + z
        z = self.pool3(z)
        x = x + z

        return x


#https://openaccess.thecvf.com/content/ICCV2021/papers/Ranftl_Vision_Transformers_for_Dense_Prediction_ICCV_2021_paper.pdf
#The add read option
class MapToken(nn.Module):
    def __init__(self, emb_size, patches=25):
        super(MapToken, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(emb_size * 2, emb_size),
                                    nn.GroupNorm(8, patches),
                                    nn.ReLU(),
                                    nn.Linear(emb_size, emb_size))

    def forward(self,x):
        cls_token = x[:,0,:].unsqueeze(dim=1)
        features = x[:,1:,:]

        cls_token = repeat(cls_token, 'b p e -> b (p repeat) e', repeat=features.shape[1])


        out = torch.cat((cls_token, features), dim=2)
        out = self.layer1(out)

        return out

class VitProject(nn.Module):
    def __init__(self, num_classes, emb_size = 1024):
        super(VitProject, self).__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(emb_size, num_classes, kernel_size=1),
                                    nn.BatchNorm2d(num_classes),
                                    nn.ReLU())
        self.layer2 = nn.Sequential(nn.ConvTranspose2d(num_classes, num_classes, 3, stride=2, padding=1),
                                    nn.BatchNorm2d(num_classes),
                                    nn.ReLU())
        self.layer3 = nn.Sequential(nn.ConvTranspose2d(num_classes, num_classes, 3, stride=2, padding=1),
                                    nn.BatchNorm2d(num_classes),
                                    nn.ReLU())

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = f.interpolate(x, (25, 25))
        return x

class DenseVitPredict(nn.Module):
    def __init__(self, num_classes):
        super(DenseVitPredict, self).__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(num_classes, num_classes, kernel_size=1),
                                    nn.GroupNorm(16, num_classes),
                                    nn.ReLU())
        self.layer2 = nn.Sequential(nn.Conv2d(num_classes, num_classes, kernel_size=1),
                                     nn.GroupNorm(16, num_classes),
                                    nn.ReLU())
        self.layer3 = nn.Sequential(nn.Conv2d(num_classes, num_classes, kernel_size=1, bias=False),
                                     nn.GroupNorm(16, num_classes, affine=False))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        return x

class VitPredict(nn.Module):
    def __init__(self, num_classes):
        super(VitPredict, self).__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(num_classes, num_classes, kernel_size=1),
                                    nn.BatchNorm2d(num_classes),
                                    nn.ReLU())
        self.layer2 = nn.Conv2d(num_classes, num_classes, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)

        return x

    

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
        self.layer2 = nn.Linear(num_channels//4, num_classes, bias=False)
        
        
    def forward(self, x):
        
        # z = x[:,shape[1]-1, :].unsqueeze(1)
        
        x = self.layer1(x)
        x = self.layer2(x)

        return x

class DePatch(nn.Module):
    def __init__(self, num_channels=750):
        super(DePatch, self).__init__()
        #self.layer0 = nn.Linear(num_channels, 60, bias=False)
        self.layer1 = Rearrange('b (h w) c -> b c h w', h=5, w=5)
        self.layer2 = nn.Upsample(scale_factor=5, mode='bilinear')
        self.layer3 = nn.Softmax(1)

    def forward(self, x):
        #x= self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

class SiamLoss(nn.Module):
    def __init__(self):
        super(SiamLoss, self).__init__()
    
    def forward(self, x, y):
        a = x/torch.linalg.norm(x, dim=1).unsqueeze(dim=1)
        b = y/torch.linalg.norm(y, dim=1).unsqueeze(dim=1)
        return -(a*b).sum(dim=1).mean()


class VitSiamLoss(nn.Module):
    def __init__(self):
        super(VitSiamLoss, self).__init__()
    
    def forward(self, x, y):
        a = x/torch.linalg.norm(x)
        b = y/torch.linalg.norm(y)
        return -(a*b).sum().mean()




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

        return p_1


class C_Enc(nn.Module):
    def __init__(self, num_channels):
        super(C_Enc, self).__init__()
        self.layer_enc = net_gen.ResnetEncoder(num_channels, num_channels, n_blocks=9)
        self.project = nn.Sequential(Rearrange('b c h w -> b (h w) c'),
                                    SegLinearUp(b1=25, b2=25, drop_class=False))
        self.predict = SegDecoder(num_channels = 25 * num_channels, patches=25)

    def forward(self, x):
        x_1 = self.layer_enc(x)

        return x_1


class C_Proj(nn.Module):
    def __init__(self, num_channels):
        super(C_Proj, self).__init__()
        self.project = nn.Sequential(Rearrange('b c h w -> b (h w) c'),
                                    SegLinearUp(b1=25, b2=25, drop_class=False))
        self.predict = SegDecoder(num_channels = 25 * num_channels, patches=25)

    def forward(self, x):
        z_1 = self.project(x)
        p_1 = self.predict(z_1)

        return p_1

       


class C_Dec(nn.Module):
    def __init__(self, num_channels):
        super(C_Dec, self).__init__()
        self.layer_dec = net_gen.ResnetDecoder(256, 60)

    def forward(self, x):
        out = self.layer_dec(x)
        return out






if __name__ == "__main__":
    s = UpsamplePredictor(256, 4)
    p = torch.ones(1, 256, 5, 5)
    q = s(p)
    print(q.shape)



