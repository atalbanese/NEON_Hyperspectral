from unittest.mock import patch
from dataloaders import RenderedDataLoader
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import models
from pytorch_lightning.callbacks import ModelCheckpoint, StochasticWeightAveraging, BaseFinetuning
import inference
import torch
import warnings
import torchvision.transforms.functional as tf
import numpy as np

import os
from pytorch_lightning import loggers as pl_loggers
from einops import rearrange
import matplotlib.pyplot as plt
import rasterio as rs
from sklearn.preprocessing import StandardScaler

class FeatureExtractorFreezeUnfreeze(BaseFinetuning):
     def __init__(self, unfreeze_at_epoch=100):
         super().__init__()
         self._unfreeze_at_epoch = unfreeze_at_epoch

     def freeze_before_training(self, pl_module):
         # freeze any module you want
         # Here, we are freezing `feature_extractor`
         self.freeze(pl_module.swav)

     def finetune_function(self, pl_module, current_epoch, optimizer, optimizer_idx):
         # When `current_epoch` is 10, feature_extractor will start training.
         if current_epoch == self._unfreeze_at_epoch:
             self.unfreeze_and_add_param_group(
                 modules=pl_module.swav,
                 optimizer=optimizer,
                 train_bn=True,
                 lr=5e-5
             )


def unified_training(class_key,
                    base_lr,
                    refine_lr,
                    class_weights,
                    features_dict,
                    num_intermediate_classes,
                    pre_train_folder,
                    train_folder,
                    valid_folder,
                    test_folder,
                    pre_training_epochs,
                    refine_epochs,
                    pre_train_batch_size,
                    refine_batch_size,
                    pre_train_workers,
                    refine_workers,
                    log_dir,
                    swa=False,
                    extra_labels="",
                    pre_training=True,
                    mode='default',
                    augment_refine=False,
                    scheduler=True, 
                    initial_freeze=None,
                    positions=False,
                    input_size=40,
                    patch_size=10,
                    full_plots=False,
                    acc_grad=1,
                    emb_size=256,
                    scaling=True,
                    augment_bright=False,
                    filters={}):

    pl.seed_everything(42)
    feature_labels = ""
    for key, value in features_dict.items():
        feature_labels = feature_labels + f"{key}_{value}_"

    refine_callback = ModelCheckpoint(
        dirpath='ckpts/', 
        filename=f'niwo_refine_{feature_labels}{extra_labels}'+'{ova:.2f}_{epoch}',
        #every_n_epochs=log_every,
        monitor='ova',
        save_on_train_epoch_end=True,
        mode='max',
        save_top_k = 3
        )

    pre_train_callback = ModelCheckpoint(
        dirpath='ckpts/', 
        filename=f'niwo_pre_train_{feature_labels}{extra_labels}'+'{epoch}',
        every_n_epochs=pre_training_epochs,
        save_on_train_epoch_end=True,
        save_top_k = -1
        )
    
    
    pre_train_ckpt = f'ckpts/niwo_pre_train_{feature_labels}{extra_labels}epoch={pre_training_epochs-1}.ckpt'

    pre_model = models.SwaVModelUnified(class_key,
                                    base_lr,
                                    class_weights,
                                    features_dict,
                                    num_intermediate_classes,
                                    pre_training=True,
                                    mode=mode,
                                    augment_refine=augment_refine,
                                    scheduler=scheduler,
                                    positions=positions,
                                    emb_size=emb_size,
                                    filters=filters,
                                    patch_size=patch_size
                                   )


    if pre_training:
        pre_train_data = RenderedDataLoader(pre_train_folder, features_dict, input_size=input_size, full_plots=full_plots)
        pre_train_loader = DataLoader(pre_train_data, batch_size=pre_train_batch_size, num_workers=pre_train_workers, pin_memory=True)

    train_dataset = RenderedDataLoader(train_folder, features_dict, stats_loc=os.path.join(pre_train_folder, 'stats/stats.npy'), input_size=input_size, full_plots=full_plots, scaling=scaling)
    train_loader = DataLoader(train_dataset, batch_size=refine_batch_size, num_workers=refine_workers)

    valid_dataset = RenderedDataLoader(valid_folder, features_dict, stats_loc=os.path.join(pre_train_folder, 'stats/stats.npy'), input_size=input_size, full_plots=full_plots, scaling=scaling)
    valid_loader = DataLoader(valid_dataset, batch_size=1, num_workers=refine_workers)

    test_dataset = RenderedDataLoader(test_folder, features_dict, stats_loc=os.path.join(pre_train_folder, 'stats/stats.npy'), input_size=input_size, full_plots=full_plots, scaling=scaling)
    test_loader = DataLoader(test_dataset, batch_size=1, num_workers=refine_workers)

    tb_logger = pl_loggers.TensorBoardLogger(save_dir=log_dir)

    if pre_training:
        pre_trainer = pl.Trainer(accelerator="gpu", max_epochs=pre_training_epochs, callbacks=[pre_train_callback], logger=tb_logger)
        
        pre_trainer.fit(pre_model, pre_train_loader)

    refine_callbacks = [refine_callback]

    if swa is not None:
        refine_callbacks.append(StochasticWeightAveraging(swa_epoch_start=swa))

    if initial_freeze is not None:
        refine_callbacks.append(FeatureExtractorFreezeUnfreeze(initial_freeze))

    
    if refine_epochs > 0:
        refiner = pl.Trainer(accelerator="gpu", max_epochs=refine_epochs, callbacks=refine_callbacks, logger=tb_logger, log_every_n_steps=1, accumulate_grad_batches=acc_grad, auto_lr_find=False)

        if pre_training_epochs > 0:
            refine_model = models.SwaVModelUnified.load_from_checkpoint(pre_train_ckpt, 
                                                                        pre_training=False, 
                                                                        mode=mode, 
                                                                        lr=refine_lr,
                                                                        augment_refine=augment_refine,
                                                                        scheduler=scheduler,
                                                                        emb_size=emb_size)
        else:
            refine_model = models.SwaVModelUnified(class_key,
                                        refine_lr,
                                        class_weights,
                                        features_dict,
                                        num_intermediate_classes,
                                        pre_training=False,
                                        mode=mode,
                                        augment_refine=augment_refine,
                                        scheduler=scheduler,
                                        positions=positions,
                                        emb_size=emb_size,
                                        augment_bright=augment_bright,
                                        filters=filters,
                                        patch_size=patch_size
                                    )
        #refiner.tune(refine_model, train_dataloaders=train_loader)
        refiner.fit(refine_model, train_dataloaders=train_loader, val_dataloaders=valid_loader)
        refiner.save_checkpoint(f'ckpts/niwo_{feature_labels}{extra_labels}_{refine_epochs}'+'.ckpt')
        refiner.test(refine_model, dataloaders=test_loader, ckpt_path='best')


def infer(ckpt, img_loc, stats_loc, block_size=128, patch_size=16, stats_key='pca', img_size=8192):
    assert block_size % patch_size == 0, 'Block size must be divisible by patch_size'
    if ".npy" in img_loc:
        img = np.load(img_loc).astype(np.float32)
        img = rearrange(img, 'h w c -> c h w')
    if ".tif" in img_loc:
        img_reader = rs.open(img_loc)
        img = img_reader.read().astype(np.float32)
        img = img[..., 0:img_size, 0:img_size]
        #img = torch.from_numpy(img)

    stats = torch.load(stats_loc)
    
    scaler = StandardScaler()
    cur_stats = stats[stats_key]
    scaler.scale_ = cur_stats['scale']
    scaler.mean_ = cur_stats['mean']
    scaler.var_ = cur_stats['var']

    img = rearrange(img, 'c h w -> (h w) c')
    img = scaler.transform(img)
    img = rearrange(img, '(h w) c -> c h w', h=img_size, w=img_size)
    img = torch.from_numpy(img).float()
    

   
    plt.imshow(rearrange(img, 'c h w -> h w c')[...,0:3])
    plt.show()


    img = torch.stack(torch.split(img, block_size, dim=1))
    img = torch.split(img,block_size, dim=3)
    img = torch.concat(img)
    model = models.SwaVModelUnified.load_from_checkpoint(ckpt, pre_training=False).eval()

    out = model.swav.forward(img)

    out = rearrange(out, 'b (h w) c -> b c h w', h=block_size//patch_size, w=block_size//patch_size)
    # 2500 12 5 5
    out = torch.split(out, img_size//block_size, dim=0)
    out = torch.concat(out, dim=3)
    out = torch.split(out, 1, dim=0)
    out = torch.concat(out, dim=2)
    out = torch.squeeze(out)
    out = out.detach()
    out = torch.argmax(out, dim=0)

    out = out.numpy()
    #64 8 8 
    plt.imshow(out)
    plt.show()


    return out



def sentinel_training(class_key,
                    base_lr,
                    refine_lr,
                    class_weights,
                    features_dict,
                    num_intermediate_classes,
                    train_folder,
                    test_folder,
                    pre_training_epochs,
                    refine_epochs,
                    pre_train_batch_size,
                    refine_batch_size,
                    pre_train_workers,
                    refine_workers,
                    log_dir,
                    swa=False,
                    extra_labels="",
                    pre_training=True,
                    mode='default',
                    augment_refine=False,
                    scheduler=True, 
                    initial_freeze=None,
                    positions=False,
                    input_size=40,
                    patch_size=10,
                    full_plots=False,
                    acc_grad=1,
                    emb_size=256,
                    scaling=True,
                    augment_bright=False,
                    filters={}):
    return None


if __name__ == "__main__":

    infer('ckpts/niwo_pre_train_rgb_3_rgb_swav_test_4epoch=9.ckpt', 
    'C:/Users/tonyt/Documents/Research/datasets/rgb/NEON.D13.NIWO.DP3.30010.001.2020-08.basic.20220814T183511Z.RELEASE-2022/2020_NIWO_4_451000_4432000_image.tif', 
    'C:/Users/tonyt/Documents/Research/datasets/tensors/rgb_blocks/stats/stats.npy',
    stats_key='rgb')
#TODO: Filters causes problems with validation loss
#8/9/10/11/12/13/14/15
    # unified_training(class_key={'PIEN': 0, 'ABLAL': 1, 'PIFL2': 2, 'PICOL': 3},
    #                 base_lr=5e-5,
    #                 refine_lr=5e-5,
    #                 #class_weights=[1.61428571, 0.60752688, 0.7739726,  2.26],
    #                 class_weights=None,
    #                 features_dict={
    #                     'rgb': 3
    #                 },
    #                 num_intermediate_classes = 4,
    #                 pre_train_folder = 'C:/Users/tonyt/Documents/Research/datasets/tensors/rgb_blocks',
    #                 train_folder = 'C:/Users/tonyt/Documents/Research/datasets/tensors/rf_test/pca_plot_split/train',
    #                 valid_folder = 'C:/Users/tonyt/Documents/Research/datasets/tensors/rf_test/pca_plot_split/test',
    #                 test_folder  = 'C:/Users/tonyt/Documents/Research/datasets/tensors/rf_test/pca_plot_split/test',
    #                 pre_training_epochs = 10,
    #                 refine_epochs = 0,
    #                 pre_train_batch_size = 256,
    #                 refine_batch_size = 64,
    #                 pre_train_workers = 4,
    #                 refine_workers = 1,
    #                 log_dir='final_logs/',
    #                 pre_training=True,
    #                 extra_labels='rgb_swav_test_4',
    #                 swa=None,
    #                 mode='patch',
    #                 scheduler=True,
    #                 initial_freeze=None,
    #                 positions=False,
    #                 emb_size=128,
    #                 patch_size=16,
    #                 scaling=True,
    #                 full_plots=True,
    #                 augment_bright=False,
    #                 )
    

# #0
#     unified_training(class_key={'PIEN': 0, 'ABLAL': 1, 'PIFL2': 2, 'PICOL': 3},
#                     base_lr=5e-5,
#                     refine_lr=5e-5,
#                     #class_weights=[1.61428571, 0.60752688, 0.7739726,  2.26],
#                     class_weights=None,
#                     features_dict={
#                         'pca': 16
#                     },
#                     num_intermediate_classes = 256,
#                     pre_train_folder = 'C:/Users/tonyt/Documents/Research/datasets/tensors/niwo_2020_pca_blocks/raw_training',
#                     train_folder = 'C:/Users/tonyt/Documents/Research/datasets/tensors/rf_test/pca_plot_split/train',
#                     valid_folder = 'C:/Users/tonyt/Documents/Research/datasets/tensors/rf_test/pca_plot_split/test',
#                     test_folder  = 'C:/Users/tonyt/Documents/Research/datasets/tensors/rf_test/pca_plot_split/test',
#                     pre_training_epochs = 0,
#                     refine_epochs = 150,
#                     pre_train_batch_size = 1024,
#                     refine_batch_size = 64,
#                     pre_train_workers = 4,
#                     refine_workers = 1,
#                     log_dir='final_logs/',
#                     pre_training=False,
#                     extra_labels='refine_test_plot',
#                     swa=None,
#                     mode='patch',
#                     scheduler=True,
#                     initial_freeze=None,
#                     positions=False,
#                     emb_size=128,
#                     patch_size=4,
#                     scaling=True,
#                     full_plots=False,
#                     augment_bright=False,
#                     )

#     #1
#     unified_training(class_key={'PIEN': 0, 'ABLAL': 1, 'PIFL2': 2, 'PICOL': 3},
#                     base_lr=5e-5,
#                     refine_lr=5e-5,
#                     #class_weights=[1.61428571, 0.60752688, 0.7739726,  2.26],
#                     class_weights=None,
#                     features_dict={
#                         'pca': 16
#                     },
#                     num_intermediate_classes = 256,
#                     pre_train_folder = 'C:/Users/tonyt/Documents/Research/datasets/tensors/niwo_2020_pca_blocks/raw_training',
#                     train_folder = 'C:/Users/tonyt/Documents/Research/datasets/tensors/rf_test/pca_object_split/train',
#                     valid_folder = 'C:/Users/tonyt/Documents/Research/datasets/tensors/rf_test/pca_object_split/test',
#                     test_folder  = 'C:/Users/tonyt/Documents/Research/datasets/tensors/rf_test/pca_object_split/test',
#                     pre_training_epochs = 0,
#                     refine_epochs = 150,
#                     pre_train_batch_size = 1024,
#                     refine_batch_size = 64,
#                     pre_train_workers = 4,
#                     refine_workers = 1,
#                     log_dir='final_logs/',
#                     pre_training=False,
#                     extra_labels='refine_test_object',
#                     swa=None,
#                     mode='patch',
#                     scheduler=True,
#                     initial_freeze=None,
#                     positions=False,
#                     emb_size=128,
#                     patch_size=4,
#                     scaling=True,
#                     full_plots=False,
#                     augment_bright=False,
#                     )
#   #2
#     unified_training(class_key={'PIEN': 0, 'ABLAL': 1, 'PIFL2': 2, 'PICOL': 3},
#                     base_lr=5e-5,
#                     refine_lr=5e-5,
#                     #class_weights=[1.61428571, 0.60752688, 0.7739726,  2.26],
#                     class_weights=None,
#                     features_dict={
#                         'pca': 16
#                     },
#                     num_intermediate_classes = 256,
#                     pre_train_folder = 'C:/Users/tonyt/Documents/Research/datasets/tensors/niwo_2020_pca_blocks/raw_training',
#                     train_folder = 'C:/Users/tonyt/Documents/Research/datasets/tensors/rf_test/pca_plot_split/train',
#                     valid_folder = 'C:/Users/tonyt/Documents/Research/datasets/tensors/rf_test/pca_plot_split/test',
#                     test_folder  = 'C:/Users/tonyt/Documents/Research/datasets/tensors/rf_test/pca_plot_split/test',
#                     pre_training_epochs = 0,
#                     refine_epochs = 150,
#                     pre_train_batch_size = 1024,
#                     refine_batch_size = 64,
#                     pre_train_workers = 4,
#                     refine_workers = 1,
#                     log_dir='final_logs/',
#                     pre_training=False,
#                     extra_labels='refine_test_plot_filters',
#                     swa=None,
#                     mode='patch',
#                     scheduler=True,
#                     initial_freeze=None,
#                     positions=False,
#                     emb_size=128,
#                     patch_size=4,
#                     scaling=True,
#                     full_plots=False,
#                     augment_bright=False,
#                     filters={
#             'shadow': 0.03,
#             'ndvi': 0.1,
#         }
#                     )
# #3
#     unified_training(class_key={'PIEN': 0, 'ABLAL': 1, 'PIFL2': 2, 'PICOL': 3},
#                     base_lr=5e-5,
#                     refine_lr=5e-5,
#                     #class_weights=[1.61428571, 0.60752688, 0.7739726,  2.26],
#                     class_weights=None,
#                     features_dict={
#                         'pca': 16
#                     },
#                     num_intermediate_classes = 256,
#                     pre_train_folder = 'C:/Users/tonyt/Documents/Research/datasets/tensors/niwo_2020_pca_blocks/raw_training',
#                     train_folder = 'C:/Users/tonyt/Documents/Research/datasets/tensors/rf_test/pca_object_split/train',
#                     valid_folder = 'C:/Users/tonyt/Documents/Research/datasets/tensors/rf_test/pca_object_split/test',
#                     test_folder  = 'C:/Users/tonyt/Documents/Research/datasets/tensors/rf_test/pca_object_split/test',
#                     pre_training_epochs = 0,
#                     refine_epochs = 150,
#                     pre_train_batch_size = 1024,
#                     refine_batch_size = 64,
#                     pre_train_workers = 4,
#                     refine_workers = 1,
#                     log_dir='final_logs/',
#                     pre_training=False,
#                     extra_labels='refine_test_object_filters',
#                     swa=None,
#                     mode='patch',
#                     scheduler=True,
#                     initial_freeze=None,
#                     positions=False,
#                     emb_size=128,
#                     patch_size=4,
#                     scaling=True,
#                     full_plots=False,
#                     augment_bright=False,
#                     filters={
#             'shadow': 0.03,
#             'ndvi': 0.1,
#         }
#                     )
# #4
#     unified_training(class_key={'PIEN': 0, 'ABLAL': 1, 'PIFL2': 2, 'PICOL': 3},
#                     base_lr=5e-5,
#                     refine_lr=5e-5,
#                     #class_weights=[1.61428571, 0.60752688, 0.7739726,  2.26],
#                     class_weights=None,
#                     features_dict={
#                         'pca': 16
#                     },
#                     num_intermediate_classes = 256,
#                     pre_train_folder = 'C:/Users/tonyt/Documents/Research/datasets/tensors/niwo_2020_pca_blocks/raw_training',
#                     train_folder = 'C:/Users/tonyt/Documents/Research/datasets/tensors/rf_test/pca_plot_split/train',
#                     valid_folder = 'C:/Users/tonyt/Documents/Research/datasets/tensors/rf_test/pca_plot_split/test',
#                     test_folder  = 'C:/Users/tonyt/Documents/Research/datasets/tensors/rf_test/pca_plot_split/test',
#                     pre_training_epochs =15,
#                     refine_epochs = 150,
#                     pre_train_batch_size = 1024,
#                     refine_batch_size = 64,
#                     pre_train_workers = 4,
#                     refine_workers = 1,
#                     log_dir='final_logs/',
#                     pre_training=True,
#                     extra_labels='refine_test_plot_pre',
#                     swa=None,
#                     mode='patch',
#                     scheduler=True,
#                     initial_freeze=None,
#                     positions=False,
#                     emb_size=128,
#                     patch_size=4,
#                     scaling=True,
#                     full_plots=False,
#                     augment_bright=False,
                    
#                     )
# #5
#     unified_training(class_key={'PIEN': 0, 'ABLAL': 1, 'PIFL2': 2, 'PICOL': 3},
#                     base_lr=5e-5,
#                     refine_lr=5e-5,
#                     #class_weights=[1.61428571, 0.60752688, 0.7739726,  2.26],
#                     class_weights=None,
#                     features_dict={
#                         'pca': 16
#                     },
#                     num_intermediate_classes = 256,
#                     pre_train_folder = 'C:/Users/tonyt/Documents/Research/datasets/tensors/niwo_2020_pca_blocks/raw_training',
#                     train_folder = 'C:/Users/tonyt/Documents/Research/datasets/tensors/rf_test/pca_object_split/train',
#                     valid_folder = 'C:/Users/tonyt/Documents/Research/datasets/tensors/rf_test/pca_object_split/test',
#                     test_folder  = 'C:/Users/tonyt/Documents/Research/datasets/tensors/rf_test/pca_object_split/test',
#                     pre_training_epochs = 15,
#                     refine_epochs = 150,
#                     pre_train_batch_size = 1024,
#                     refine_batch_size = 64,
#                     pre_train_workers = 4,
#                     refine_workers = 1,
#                     log_dir='final_logs/',
#                     pre_training=True,
#                     extra_labels='refine_test_object_pre',
#                     swa=None,
#                     mode='patch',
#                     scheduler=True,
#                     initial_freeze=None,
#                     positions=False,
#                     emb_size=128,
#                     patch_size=4,
#                     scaling=True,
#                     full_plots=False,
#                     augment_bright=False,
#                     )
# #6
#     unified_training(class_key={'PIEN': 0, 'ABLAL': 1, 'PIFL2': 2, 'PICOL': 3},
#                     base_lr=5e-5,
#                     refine_lr=5e-5,
#                     #class_weights=[1.61428571, 0.60752688, 0.7739726,  2.26],
#                     class_weights=None,
#                     features_dict={
#                         'pca': 16
#                     },
#                     num_intermediate_classes = 256,
#                     pre_train_folder = 'C:/Users/tonyt/Documents/Research/datasets/tensors/niwo_2020_pca_blocks/raw_training',
#                     train_folder = 'C:/Users/tonyt/Documents/Research/datasets/tensors/rf_test/pca_plot_split/train',
#                     valid_folder = 'C:/Users/tonyt/Documents/Research/datasets/tensors/rf_test/pca_plot_split/test',
#                     test_folder  = 'C:/Users/tonyt/Documents/Research/datasets/tensors/rf_test/pca_plot_split/test',
#                     pre_training_epochs =15,
#                     refine_epochs = 150,
#                     pre_train_batch_size = 1024,
#                     refine_batch_size = 64,
#                     pre_train_workers = 4,
#                     refine_workers = 1,
#                     log_dir='final_logs/',
#                     pre_training=True,
#                     extra_labels='refine_test_plot_pre_filters',
#                     swa=None,
#                     mode='patch',
#                     scheduler=True,
#                     initial_freeze=None,
#                     positions=False,
#                     emb_size=128,
#                     patch_size=4,
#                     scaling=True,
#                     full_plots=False,
#                     augment_bright=False,
#                     filters={
#             'shadow': 0.03,
#             'ndvi': 0.1,
#         }
#                     )
# #7
#     unified_training(class_key={'PIEN': 0, 'ABLAL': 1, 'PIFL2': 2, 'PICOL': 3},
#                     base_lr=5e-5,
#                     refine_lr=5e-5,
#                     #class_weights=[1.61428571, 0.60752688, 0.7739726,  2.26],
#                     class_weights=None,
#                     features_dict={
#                         'pca': 16
#                     },
#                     num_intermediate_classes = 256,
#                     pre_train_folder = 'C:/Users/tonyt/Documents/Research/datasets/tensors/niwo_2020_pca_blocks/raw_training',
#                     train_folder = 'C:/Users/tonyt/Documents/Research/datasets/tensors/rf_test/pca_object_split/train',
#                     valid_folder = 'C:/Users/tonyt/Documents/Research/datasets/tensors/rf_test/pca_object_split/test',
#                     test_folder  = 'C:/Users/tonyt/Documents/Research/datasets/tensors/rf_test/pca_object_split/test',
#                     pre_training_epochs = 15,
#                     refine_epochs = 150,
#                     pre_train_batch_size = 1024,
#                     refine_batch_size = 64,
#                     pre_train_workers = 4,
#                     refine_workers = 1,
#                     log_dir='final_logs/',
#                     pre_training=True,
#                     extra_labels='refine_test_object_pre_filters',
#                     swa=None,
#                     mode='patch',
#                     scheduler=True,
#                     initial_freeze=None,
#                     positions=False,
#                     emb_size=128,
#                     patch_size=4,
#                     scaling=True,
#                     full_plots=False,
#                     augment_bright=False,
#                     filters={
#             'shadow': 0.03,
#             'ndvi': 0.1,
#         }
#                     )
    
    

    # infer(
    #     #'ckpts/niwo_pre_train_pca_16_small_embedepoch=1.ckpt', 
    #     'ckpts/niwo_pre_train_pca_16_test_for_viz_manyepoch=99.ckpt',
    # 'C:/Users/tonyt/Documents/Research/datasets/pca/niwo_16_unmasked/NEON_D13_NIWO_DP3_451000_4431000_reflectance_pca.npy', 
    # 'C:/Users/tonyt/Documents/Research/datasets/tensors/niwo_2020_pca_blocks/raw_training/stats/stats.npy',
    # block_size=100)

    # infer('ckpts/niwo_pre_train_pca_16_test_for_viz_many_augepoch=14.ckpt', 
    # 'C:/Users/tonyt/Documents/Research/datasets/pca/niwo_16_unmasked/NEON_D13_NIWO_DP3_451000_4433000_reflectance_pca.npy', 
    # 'C:/Users/tonyt/Documents/Research/datasets/tensors/niwo_2020_pca_blocks/raw_training/stats/stats.npy')

    # infer('ckpts/niwo_pre_train_nmf_20_test_for_nmf_many_augepoch=24.ckpt', 
    # 'C:/Users/tonyt/Documents/Research/datasets/pca/niwo_mnf/NEON_D13_NIWO_DP3_451000_4433000_reflectance_pca.npy', 
    # 'C:/Users/tonyt/Documents/Research/datasets/tensors/niwo_2020_nmf_blocks/raw_training/stats/stats.npy')
