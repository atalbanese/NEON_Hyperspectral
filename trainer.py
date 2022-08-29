from dataloaders import RenderedDataLoader
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import models
from pytorch_lightning.callbacks import ModelCheckpoint, StochasticWeightAveraging, BaseFinetuning
import inference
import torch
import warnings
import torchvision.transforms.functional as tf
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
                    filters=[]):

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
                                        patch_size=input_size
                                    )
        #refiner.tune(refine_model, train_dataloaders=train_loader)
        refiner.fit(refine_model, train_dataloaders=train_loader, val_dataloaders=valid_loader)
        refiner.save_checkpoint(f'ckpts/niwo_{feature_labels}{extra_labels}_{refine_epochs}'+'.ckpt')
        refiner.test(refine_model, dataloaders=test_loader, ckpt_path='best')


def rgb_infer(ckpt, img_loc, stats_loc):
    img_reader = rs.open(img_loc)
    img = img_reader.read()

    if os.path.exists(os.path.join(stats_loc)):
        stats = torch.load(stats_loc)
    
    scaler = StandardScaler()
    cur_stats = stats['rgb']
    scaler.scale_ = cur_stats['scale']
    scaler.mean_ = cur_stats['mean']
    scaler.var_ = cur_stats['var']

    img = rearrange(img, 'c h w -> (h w) c')
    img = scaler.transform(img)
    img = rearrange(img, '(h w) c -> c h w', h=10000, w=10000)
    img = torch.from_numpy(img).float()
    
    #img = tf.center_crop(img, [512,512])
    img = torch.nn.functional.interpolate(img.unsqueeze(0), scale_factor=0.1).squeeze()
    plt.imshow(rearrange(img, 'c h w -> h w c'))
    plt.show()

    test = torch.argmax(img, dim=0)
    plt.imshow(test)
    img = img.unsqueeze(0)

    # # 3 512 512
    # img = torch.stack(torch.split(img, 64, dim=1))
    # # 8 3 64 512
    # img = torch.split(img,64, dim=3)
    # img = torch.concat(img)
    # #64 3 64 64
    model = models.SwaVModelUnified.load_from_checkpoint(ckpt, pre_training=False).eval()

    out = model.swav.forward(img)

    out = out.squeeze()
    out = torch.argmax(out, dim=0)

    # out = torch.argmax(out, dim=2)
    #out = rearrange(out, 'b (h w) -> b h w', h=8, w=8)
    #out = out.squeeze()
    out = out.detach()
    # out = torch.split(out, 8, dim=0)
    # out = torch.concat(out, dim=2)
    # out = torch.split(out, 1, dim=0)
    # out = torch.concat(out, dim=1)
    # out = torch.squeeze(out)

    out = out.numpy()
    #64 8 8 
    plt.imshow(out)
    plt.show()


    return out






if __name__ == "__main__":

    #rgb_infer('ckpts/niwo_pre_train_rgb_3_rgb_unsup_unet_no_scaleepoch=0.ckpt', 'C:/Users/tonyt/Documents/Research/datasets/rgb/NEON.D13.NIWO.DP3.30010.001.2020-08.basic.20220814T183511Z.RELEASE-2022/2020_NIWO_4_451000_4432000_image.tif', 'C:/Users/tonyt/Documents/Research/datasets/tensors/rgb_blocks/stats/stats.npy')


    unified_training(class_key={'PIEN': 0, 'ABLAL': 1, 'PIFL2': 2, 'PICOL': 3},
                    base_lr=5e-5,
                    refine_lr=.00013,
                    #class_weights=[1.61428571, 0.60752688, 0.7739726,  2.26],
                    class_weights=None,
                    features_dict={
                        'rgb': 3
                    },
                    num_intermediate_classes = 10,
                    pre_train_folder = 'C:/Users/tonyt/Documents/Research/datasets/tensors/rgb_blocks',
                    train_folder = 'C:/Users/tonyt/Documents/Research/datasets/tensors/niwo_2020_pca_blocks/tt_train_hs',
                    valid_folder = 'C:/Users/tonyt/Documents/Research/datasets/tensors/niwo_2020_pca_blocks/tt_valid_hs',
                    test_folder  = 'C:/Users/tonyt/Documents/Research/datasets/tensors/niwo_2020_pca_blocks/tt_test_hs',
                    pre_training_epochs = 2,
                    refine_epochs = 0,
                    pre_train_batch_size = 128,
                    refine_batch_size = 8,
                    pre_train_workers = 4,
                    refine_workers = 1,
                    log_dir='exp_logs/',
                    pre_training=True,
                    extra_labels='rgb_unsup_unet_no_scale',
                    swa=None,
                    mode='patch',
                    scheduler=True,
                    initial_freeze=None,
                    positions=False,
                    emb_size=256,
                    patch_size=8,
                    scaling=False,
                    full_plots=True,
                    augment_bright=False,
                    filters=None)