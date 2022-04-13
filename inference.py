import pytorch_lightning as pl
import models
import h5_helper as hp
import utils as utils
import matplotlib.pyplot as plt
import numpy as np
import torch
from einops import rearrange, reduce, repeat

def load_ckpt(model_class, ckpt, **kwargs):
    model= model_class(**kwargs).load_from_checkpoint(ckpt, **kwargs)
    return model.eval()

def do_inference(model, file, reshape, pca, **kwargs):
    if not pca:
        opened = hp.pre_processing(file, get_all=True)["bands"]
        reduce_dim = utils.pca(opened, True, **kwargs)
    else:
        reduce_dim = np.load(file)
    if reshape:
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

def transformer_inshape(inp):
    #inp = inp[:, 0:999, 0:999]
    batched = rearrange(inp, "c (b1 h) (b2 w) -> (b1 b2) c h w", h=25, w=25, b1=40, b2=40)
    return batched

def transformer_outshape(inp):
    unbatched = rearrange(inp, "(b1 b2) h w -> (b1 h) (b2 w)", h=25, w=25, b1=40, b2=40)
    return unbatched


    


if __name__ == "__main__":
    h5_fold = "/data/shared/src/aalbanese/datasets/hs/NEON_refl-surf-dir-ortho-mosaic/NEON.D16.WREF.DP3.30006.001.2021-07.basic.20220330T192306Z.PROVISIONAL"
    ckpt = "/data/shared/src/aalbanese/lidar_hs_unsup_dl_model/lightning_logs/version_59/checkpoints/epoch=20-step=4494.ckpt"
    h5_file = "/data/shared/src/aalbanese/datasets/hs/NEON_refl-surf-dir-ortho-mosaic/NEON.D16.WREF.DP3.30006.001.2021-07.basic.20220330T192306Z.PROVISIONAL/NEON_D16_WREF_DP3_580000_5075000_reflectance.h5"

    test = torch.ones(1369, 1, 27, 27)
    print(transformer_outshape(test).shape)




