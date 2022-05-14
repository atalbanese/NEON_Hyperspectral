import rasterio as rs
import pytorch_lightning as pl
import models
import h5_helper as hp
import utils as utils
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as f
import torch
from einops import rearrange, reduce, repeat

def load_ckpt(model_class, ckpt, **kwargs):
    model= model_class(**kwargs).load_from_checkpoint(ckpt, **kwargs)
    return model.eval()

def do_inference(model, file, reshape, pca, norm, **kwargs):
    if not pca:
        opened = hp.pre_processing(file, get_all=True)["bands"]
        reduce_dim = utils.pca(opened, True, **kwargs)
    else:
        reduce_dim = np.load(file)
        reduce_dim = norm(reduce_dim)
    if reshape:
        # if kwargs['rearrange']:
        #     reduce_dim = rearrange(reduce_dim, 'h w c -> c h w')
        reduce_dim = transformer_inshape(reduce_dim)
        reduce_dim = torch.from_numpy(reduce_dim).float()
    else:    
        reduce_dim = torch.from_numpy(reduce_dim).float().unsqueeze(0)
    y = model.forward(reduce_dim)
    y = utils.get_classifications(y).squeeze()
    y = y.detach().numpy()
    if reshape:
        y = transformer_outshape(y)
    return y

def vit_inference(model, file, norm):
    model.eval()
    img = np.load(file)
    img = img[:500,:500,...]
    img = torch.from_numpy(img).float()
    img = rearrange(img, 'h w c -> c h w')
    img = f.interpolate(img.unsqueeze(0), size=(512, 512)).squeeze(0)
    mask = img != img
    #Set nans to 0
    img[mask] = 0
    img = norm(img)
    img[mask] = 0

    #img = rearrange(img, 'c (b1 h) (b2 w) -> (b1 b2) c h w', h=32, w=32, b1=16, b2=16)
    #mask = rearrange(mask, 'c (b1 h) (b2 w) -> (b1 b2) c h w', h=32, w=32, b1=16, b2=16)

    img = img.unsqueeze(0)
    mask = mask.unsqueeze(0)

    img = model(img).detach()

    mask = mask[:,0,:,:]
    img[mask] = -1
    img = rearrange(img, "(b1 b2) h w -> (b1 h) (b2 w)",  h=32, w=32, b1=16, b2=16)
    img = img.numpy()
    return img


def swav_inference_big(model, file):
    model.eval()
    img = np.load(file)
    img = torch.from_numpy(img).float()
    
    img = rearrange(img, '(k1 h) (k2 w) c -> (k1 k2) c h w', k1=2, k2=2)
    imgs = []
    #img_shape = list(img.shape)
    #img_shape[1] = 24
    #img_shape = (4, 24, 500, 500)
    #holder = torch.empty(img_shape, dtype=img.dtype)

    for i, x in enumerate(img):


        x = x.unsqueeze(0).clone()

        x = model(x).detach()
        #print(holder.shape)
        #print(x.shape)
        x = rearrange(x, 'b (h w) c -> b c h w', h=125, w=125)
        #print(x.shape)
        x = f.interpolate(x, size=(500, 500), mode='bilinear')
        #print(x.shape)
        imgs.append(torch.argmax(x.squeeze(0), dim=0).numpy())
    #img = rearrange(holder, '(k1 k2) c h w -> c (h k1) (w k2)', k1=2, k2=2)
    #img = torch.argmax(img, dim=0)
    #img = img.numpy()
    row_1 = np.hstack((imgs[0], imgs[1]))
    row_2 = np.hstack((imgs[2], imgs[3]))
    img = np.vstack((row_1,row_2))
    #plt.imshow(img, cmap='tab20')
    #plt.show()

    return img

def swav_inference_small(model, file):
    model.eval()
    img = np.load(file)
    #img = img[500:,250:750,...]
    img = torch.from_numpy(img).float()
    img = rearrange(img, 'h w c -> c h w')

    img = img.unsqueeze(0)

    img = model(img).detach()
    img = rearrange(img, 'b (h w) c -> b c h w', h=10, w=10)
    img = f.interpolate(img, size=(40, 40), mode='bilinear')
    img = torch.argmax(img, dim=1)
    img = img.squeeze().numpy()
    #plt.imshow(img)
    #plt.show()

    return img

def swav_inference_res(model, file):
    model.eval()
    img = np.load(file)
    #img = img[500:,250:750,...]
    img = torch.from_numpy(img).float()
    img = rearrange(img, 'h w c -> c h w')

    img = img.unsqueeze(0)

    img = model(img).detach()
    img = torch.argmax(img, dim=1)
    img = img.squeeze().numpy()
    plt.imshow(img, cmap='PRGn')
    plt.show()

    return img


def swav_inference_big_struct_3(model, file, chm, azm):
    model.eval()
     #PCA
    img = np.load(file).astype(np.float32)
    #img = rearrange(img, 'h w c -> c h w')
    img = torch.from_numpy(img)

    #Azimuth
    azimuth = np.load(azm).astype(np.float32)
    #Make -1 to 1
    azimuth = (torch.from_numpy(azimuth)-180)/180
    azimuth[azimuth != azimuth] = 0

    #CHM
 
    chm_open = rs.open(chm)
    chm = chm_open.read().astype(np.float32)
    #Make 0 to 1 - 47.33... is max height in dataset
    chm = torch.from_numpy(chm).squeeze(0)/47.33000183105469
    chm[chm != chm] = 0
    
    img = rearrange(img, '(k1 h) (k2 w) c -> (k1 k2) c h w', k1=2, k2=2)
    chm = rearrange(chm, '(k1 h) (k2 w) -> (k1 k2) h w', k1=2, k2=2).unsqueeze(1).unsqueeze(1)
    azm = rearrange(azimuth, '(k1 h) (k2 w) -> (k1 k2) h w', k1=2, k2=2).unsqueeze(1).unsqueeze(1)
    imgs = []
    #img_shape = list(img.shape)
    #img_shape[1] = 24
    #img_shape = (4, 24, 500, 500)
    #holder = torch.empty(img_shape, dtype=img.dtype)

    for i, x in enumerate(img):


        x = x.unsqueeze(0).clone()

        x = f.interpolate(x, size=(501, 501), mode='bilinear')
        c = f.interpolate(chm[i], size=(501, 501), mode='bilinear')
        a = f.interpolate(azm[i], size=(501, 501), mode='bilinear')


        x = model(x, c, a).detach()
        #print(holder.shape)
        #print(x.shape)
        x = rearrange(x, 'b (h w) c -> b c h w', h=167, w=167)
        #print(x.shape)
        #x = f.interpolate(x, size=(500, 500), mode='bilinear')
        #print(x.shape)
        imgs.append(x.squeeze(0))
    #img = rearrange(holder, '(k1 k2) c h w -> c (h k1) (w k2)', k1=2, k2=2)
    #img = torch.argmax(img, dim=0)
    #img = img.numpy()
    row_1 = torch.concat((imgs[0], imgs[1]), dim=2)
    row_2 = torch.concat((imgs[2], imgs[3]), dim=2)
    img = torch.concat((row_1, row_2), dim=1)
    img = f.interpolate(img.unsqueeze(0), size=(1000,1000), mode='bilinear')
    img = torch.argmax(img.squeeze(0), dim=0)
    img = img.detach().numpy()
    plt.imshow(img, cmap='tab20')
    plt.show()

    return img

def swav_inference_big_struct_4(model, file, chm, azm):
    model.eval()
     #PCA
    img = np.load(file).astype(np.float32)
    #img = rearrange(img, 'h w c -> c h w')
    img = torch.from_numpy(img)

    #Azimuth
    azimuth = np.load(azm).astype(np.float32)
    #Make -1 to 1
    azimuth = (torch.from_numpy(azimuth)-180)/180
    azimuth[azimuth != azimuth] = 0

    #CHM
 
    chm_open = rs.open(chm)
    chm = chm_open.read().astype(np.float32)
    #Make 0 to 1 - 47.33... is max height in dataset
    chm = torch.from_numpy(chm).squeeze(0)/47.33000183105469
    chm[chm != chm] = 0
    
    img = rearrange(img, '(k1 h) (k2 w) c -> (k1 k2) c h w', k1=2, k2=2)
    chm = rearrange(chm, '(k1 h) (k2 w) -> (k1 k2) h w', k1=2, k2=2).unsqueeze(1).unsqueeze(1)
    azm = rearrange(azimuth, '(k1 h) (k2 w) -> (k1 k2) h w', k1=2, k2=2).unsqueeze(1).unsqueeze(1)
    imgs = []

    for i, x in enumerate(img):


        x = x.unsqueeze(0).clone()

        x = model(x, chm[i], azm[i]).detach()
        
        x = rearrange(x, 'b (h w) c -> b c h w', h=125, w=125)

        imgs.append(x.squeeze(0))

    row_1 = torch.concat((imgs[0], imgs[1]), dim=2)
    row_2 = torch.concat((imgs[2], imgs[3]), dim=2)
    img = torch.concat((row_1, row_2), dim=1)
    img = f.interpolate(img.unsqueeze(0), size=(1000,1000), mode='bilinear')
    img = torch.argmax(img.squeeze(0), dim=0)
    img = img.detach().numpy()
    plt.imshow(img, cmap='tab20')
    plt.show()

    return img




def transformer_inshape(inp):
    #inp = inp[:, 0:999, 0:999]
    batched = rearrange(inp, "c (b1 h) (b2 w) -> (b1 b2) c h w", h=25, w=25, b1=40, b2=40)
    return batched

def transformer_outshape(inp):
    unbatched = rearrange(inp, "(b1 b2) h w -> (b1 h) (b2 w)", h=25, w=25, b1=40, b2=40)
    return unbatched



if __name__ == "__main__":
    # h5_fold = "/data/shared/src/aalbanese/datasets/hs/NEON_refl-surf-dir-ortho-mosaic/NEON.D16.WREF.DP3.30006.001.2021-07.basic.20220330T192306Z.PROVISIONAL"
    # ckpt = "/data/shared/src/aalbanese/lidar_hs_unsup_dl_model/lightning_logs/version_59/checkpoints/epoch=20-step=4494.ckpt"
    # h5_file = "/data/shared/src/aalbanese/datasets/hs/NEON_refl-surf-dir-ortho-mosaic/NEON.D16.WREF.DP3.30006.001.2021-07.basic.20220330T192306Z.PROVISIONAL/NEON_D16_WREF_DP3_580000_5075000_reflectance.h5"

    # test = torch.ones(1369, 1, 27, 27)
    # print(transformer_outshape(test).shape)
    ckpt = 'ckpts\harv_10_channels_12_classes_swav_structure_patch_size_4_no_struct_queue_epoch=49.ckpt'
    pca_file = 'C:/Users/tonyt/Documents/Research/datasets/pca/harv_2022_10_channels/NEON_D01_HARV_DP3_730000_4701000_reflectance_pca.npy'
    chm_file = 'C:/Users/tonyt/Documents/Research/datasets/chm/harv_2019/NEON_struct-ecosystem/NEON.D01.HARV.DP3.30015.001.2019-08.basic.20220511T165943Z.RELEASE-2022/NEON_D01_HARV_DP3_730000_4701000_CHM.tif'
    azm_file = 'C:/Users/tonyt/Documents/Research/datasets/solar_azimuth/harv_2022/NEON_D01_HARV_DP3_730000_4701000_reflectance_solar.npy'
    MODEL = models.SWaVModelStruct(patch_size=4, img_size=40, azm=False, chm=False, use_queue=True).load_from_checkpoint(ckpt, patch_size=4, img_size=40, azm=False, chm=False, use_queue=True)

    swav_inference_big_struct_4(MODEL, pca_file, chm_file, azm_file)



