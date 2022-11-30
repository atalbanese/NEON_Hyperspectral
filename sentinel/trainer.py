from operator import index
from dataloaders import SentinelDataLoader
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
import models
from pytorch_lightning.callbacks import ModelCheckpoint, StochasticWeightAveraging, BaseFinetuning
from pytorch_lightning import loggers as pl_loggers
import os
import numpy as np
import pandas as pd


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
                    num_intermediate_classes,
                    train_folder,
                    test_folder,
                    targets_folder,
                    stats_loc,
                    pre_training_epochs,
                    refine_epochs,
                    pre_train_batch_size,
                    refine_batch_size,
                    pre_train_workers,
                    refine_workers,
                    train_split = 0.8,
                    swa=None,
                    extra_labels="",
                    pre_training=True,
                    augment_refine=False,
                    scheduler=True, 
                    initial_freeze=None,
                    positions=False,
                    patch_size=4,
                    acc_grad=1,
                    emb_size=256,
                    augment_bright=False,
                    crop_size=None
                    ):

    pl.seed_everything(42)

    refine_callback = ModelCheckpoint(
        dirpath='ckpts/', 
        filename=f'sentinel_refine_{extra_labels}'+'{ova:.2f}_{epoch}',
        #every_n_epochs=log_every,
        monitor='ova',
        save_on_train_epoch_end=True,
        mode='max',
        save_top_k = 3
        )

    pre_train_callback = ModelCheckpoint(
        dirpath='ckpts/', 
        filename=f'sentinel_pre_train_{extra_labels}'+'{epoch}',
        every_n_epochs=pre_training_epochs,
        save_on_train_epoch_end=True,
        save_top_k = -1
        )
    
    
    pre_train_ckpt = f'ckpts/sentinel_pre_train_{extra_labels}epoch={pre_training_epochs-1}.ckpt'

    pre_model = models.SwaVModelUnified(class_key,
                                    base_lr,
                                    class_weights,
                                    num_intermediate_classes,
                                    pre_training=True,
                                    augment_refine=augment_refine,
                                    scheduler=scheduler,
                                    positions=positions,
                                    emb_size=emb_size,
                                    patch_size=patch_size
                                   )


    if pre_training:
        pre_train_data = SentinelDataLoader(train_folder, targets_folder, stats_loc, crop_size=crop_size)
        pre_train_loader = DataLoader(pre_train_data, batch_size=pre_train_batch_size, num_workers=pre_train_workers, pin_memory=True)

    all_dataset = SentinelDataLoader(train_folder, targets_folder, stats_loc, crop_size=crop_size)

    num_to_train = int(len(all_dataset) * train_split)
    train_dataset, valid_dataset = torch.utils.data.random_split(all_dataset, [num_to_train, len(all_dataset) - num_to_train], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_dataset, batch_size=refine_batch_size, num_workers=refine_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=refine_batch_size, num_workers=refine_workers)


    test_dataset = SentinelDataLoader(train_folder, test_folder, stats_loc, testing=True)
    test_loader = DataLoader(test_dataset, batch_size=1, num_workers=refine_workers)

    #tb_logger = pl_loggers.TensorBoardLogger(save_dir=log_dir)

    if pre_training:
        pre_trainer = pl.Trainer(accelerator="gpu", max_epochs=pre_training_epochs, callbacks=[pre_train_callback])
        
        pre_trainer.fit(pre_model, pre_train_loader)

    refine_callbacks = [refine_callback]

    if swa is not None:
        refine_callbacks.append(StochasticWeightAveraging(swa_epoch_start=swa))

    if initial_freeze is not None:
        refine_callbacks.append(FeatureExtractorFreezeUnfreeze(initial_freeze))

    
    if refine_epochs > 0:
        refiner = pl.Trainer(accelerator="gpu", max_epochs=refine_epochs, callbacks=refine_callbacks, log_every_n_steps=1, accumulate_grad_batches=acc_grad, auto_lr_find=False)

        if pre_training_epochs > 0:
            refine_model = models.SwaVModelUnified.load_from_checkpoint(pre_train_ckpt, 
                                                                        pre_training=False, 
                                                                        lr=refine_lr,
                                                                        augment_refine=augment_refine,
                                                                        scheduler=scheduler,
                                                                        emb_size=emb_size)
        else:
            refine_model = models.SwaVModelUnified(class_key,
                                        refine_lr,
                                        class_weights,
                                        num_intermediate_classes,
                                        pre_training=False,
                                        augment_refine=augment_refine,
                                        scheduler=scheduler,
                                        positions=positions,
                                        emb_size=emb_size,
                                        augment_bright=augment_bright,
                                        patch_size=patch_size
                                    )
        #refiner.tune(refine_model, train_dataloaders=train_loader)
        refiner.fit(refine_model, train_dataloaders=train_loader, val_dataloaders=valid_loader)
        refiner.save_checkpoint(f'ckpts/sentinel_{extra_labels}_{refine_epochs}'+'.ckpt')
        #refiner.test(refine_model, dataloaders=test_loader, ckpt_path='best')

def infer(model_ckpt, base_dir, targets_dir, stats_loc, class_key, out_file):
    model = models.SwaVModelUnified.load_from_checkpoint(model_ckpt)
    model.eval()

    dataset = SentinelDataLoader(base_dir, targets_dir, stats_loc, testing=True)
    test_loader = DataLoader(dataset, batch_size=1)

    col_labels = ['Field ID'] + list(class_key.keys())
    output_probs = []

    for x in test_loader:
        output = model(x).squeeze().detach().numpy()
        field_ids = x['field_ids']
        field_ids = field_ids.squeeze().detach().numpy()

        for id in np.unique(field_ids):
            if id != 0:
                id_mask = field_ids == id
                pixel_holder = {p:{} for p in range(0,id_mask.sum())}
                for ix in class_key.values():
                    cur_probs = output[ix]
                    cur_probs = cur_probs[id_mask]
                    for p in pixel_holder.keys():
                        pixel_holder[p][ix] = cur_probs[p]

                append_base = [id]
                for v in pixel_holder.values():
                    to_append = append_base + list(v.values())
                    output_probs.append(to_append)
        print('here')
    print('here')

    to_save = pd.DataFrame.from_records(output_probs, columns=col_labels)
    to_save = to_save.groupby('Field ID').aggregate('mean')
    to_save.to_csv(out_file)


if __name__ == '__main__':

    BASE_DIR = r'C:\Users\tonyt\Documents\agrifield\ref_agrifieldnet_competition_v1_source'
    TARGET_DIR = r'C:\Users\tonyt\Documents\agrifield\train\ref_agrifieldnet_competition_v1_labels_train'
    TEST_DIR = r'C:\Users\tonyt\Documents\agrifield\test\ref_agrifieldnet_competition_v1_labels_test'
    STATS_LOC = r'C:\Users\tonyt\Documents\agrifield\stats.npy'

    CLASS_KEY = {
        'Wheat': 0,
        'Mustard': 1,
        'Lentil': 2,
        'No Crop': 3,
        'Green Pea': 4,
        'Sugarcane': 5,
        'Garlic': 6,
        'Maize': 7,
        'Gram': 8,
        'Coriander': 9,
        'Potato': 10,
        'Bersem': 11,
        'Rice': 12
    }

    #infer(r'ckpts\sentinel_refine_ova=0.69_epoch=595.ckpt', BASE_DIR, TEST_DIR, STATS_LOC, CLASS_KEY, '')

    #reconcile(r'C:\Users\tonyt\Documents\agrifield\output_1.csv', r'C:\Users\tonyt\Documents\agrifield\submission_1_median.csv')
    #54
    # unified_training(
    #     class_key=CLASS_KEY,
    #     base_lr = 5e-5,
    #     refine_lr = 5e-4,
    #     class_weights=None,
    #     num_intermediate_classes=128,
    #     train_folder = BASE_DIR,
    #     test_folder=TEST_DIR,
    #     targets_folder=TARGET_DIR,
    #     stats_loc=STATS_LOC,
    #     pre_training_epochs = 0,
    #     refine_epochs=400,
    #     pre_train_batch_size=64,
    #     refine_batch_size=128,
    #     pre_train_workers=6,
    #     refine_workers=6,
    #     patch_size=4,
    #     pre_training=False,
    #     crop_size=64,
    #     extra_labels='No_Pre'
    # )
    # #55
    # unified_training(
    #     class_key=CLASS_KEY,
    #     base_lr = 5e-5,
    #     refine_lr = 5e-4,
    #     class_weights=None,
    #     num_intermediate_classes=64,
    #     train_folder = BASE_DIR,
    #     test_folder=TEST_DIR,
    #     targets_folder=TARGET_DIR,
    #     stats_loc=STATS_LOC,
    #     pre_training_epochs = 800,
    #     refine_epochs=400,
    #     pre_train_batch_size=64,
    #     refine_batch_size=128,
    #     pre_train_workers=6,
    #     refine_workers=6,
    #     patch_size=4,
    #     pre_training=True,
    #     crop_size=64,
    #     extra_labels='64_Int'
    # )
    #56
    unified_training(
        class_key=CLASS_KEY,
        base_lr = 5e-5,
        refine_lr = 5e-4,
        class_weights=[0.1927511,0.30926304,5.02222578,0.39780963,27.26756483,2.4878139,4.59653236,1.65041342,4.13333626,21.35557068,16.34207328,55.47539051,4.24606361],
        num_intermediate_classes=128,
        train_folder = BASE_DIR,
        test_folder=TEST_DIR,
        targets_folder=TARGET_DIR,
        stats_loc=STATS_LOC,
        pre_training_epochs = 800,
        refine_epochs=600,
        pre_train_batch_size=64,
        refine_batch_size=128,
        pre_train_workers=6,
        refine_workers=6,
        patch_size=4,
        pre_training=False,
        crop_size=64,
    )

    #50
    # unified_training(
    #     class_key=CLASS_KEY,
    #     base_lr = 5e-5,
    #     refine_lr = 5e-4,
    #     class_weights=None,
    #     num_intermediate_classes=128,
    #     train_folder = BASE_DIR,
    #     test_folder=TEST_DIR,
    #     targets_folder=TARGET_DIR,
    #     stats_loc=STATS_LOC,
    #     pre_training_epochs = 800,
    #     refine_epochs=400,
    #     pre_train_batch_size=64,
    #     refine_batch_size=128,
    #     pre_train_workers=6,
    #     refine_workers=6,
    #     patch_size=4,
    #     pre_training=False,
    #     crop_size=64
    # )
    #51
    # unified_training(
    #     class_key=CLASS_KEY,
    #     base_lr = 5e-5,
    #     refine_lr = 5e-4,
    #     class_weights=None,
    #     num_intermediate_classes=128,
    #     train_folder = BASE_DIR,
    #     test_folder=TEST_DIR,
    #     targets_folder=TARGET_DIR,
    #     stats_loc=STATS_LOC,
    #     pre_training_epochs = 800,
    #     refine_epochs=400,
    #     pre_train_batch_size=64,
    #     refine_batch_size=128,
    #     pre_train_workers=6,
    #     refine_workers=6,
    #     patch_size=4,
    #     pre_training=False,
    #     crop_size=64,
    #     initial_freeze=300
    # )
    #52
    # unified_training(
    #     class_key=CLASS_KEY,
    #     base_lr = 5e-5,
    #     refine_lr = 5e-4,
    #     class_weights=None,
    #     num_intermediate_classes=128,
    #     train_folder = BASE_DIR,
    #     test_folder=TEST_DIR,
    #     targets_folder=TARGET_DIR,
    #     stats_loc=STATS_LOC,
    #     pre_training_epochs = 800,
    #     refine_epochs=400,
    #     pre_train_batch_size=64,
    #     refine_batch_size=128,
    #     pre_train_workers=6,
    #     refine_workers=6,
    #     patch_size=4,
    #     pre_training=False,
    #     crop_size=64,
    #     swa=0.75
    # )