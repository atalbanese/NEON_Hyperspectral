from dataloaders import StructureDataset, MergedStructureDataset, RenderedDataLoader
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import models
from pytorch_lightning.callbacks import ModelCheckpoint, StochasticWeightAveraging
import inference
import warnings

#TODO: Generalize This
def do_training(num_channels=10, num_classes=12, azm=True, chm=True, patch_size=4, log_every=5, max_epochs=50, num_workers=4, img_size=40, extra_labels='', use_queue=False,  same_embed=False, concat=False, queue_chunks=1, azm_concat=False, chm_concat=False, main_brightness=False, aug_brightness=False, rescale_pca=False):
    pca_fold = 'C:/Users/tonyt/Documents/Research/datasets/pca/niwo_10_channels'
    chm_fold = 'C:/Users/tonyt/Documents/Research/datasets/chm/niwo/'
    az_fold = 'C:/Users/tonyt/Documents/Research/datasets/solar_azimuth/niwo'

    checkpoint_callback = ModelCheckpoint(
        dirpath='ckpts', 
        filename=f'niwo_{num_channels}_channels_{num_classes}_classes_swav_structure_patch_size_{patch_size}_{extra_labels}'+'_{epoch}',
        every_n_epochs=log_every,
        save_on_train_epoch_end=True,
        save_top_k = -1
        )

    pl.seed_everything(42)

    dataset = StructureDataset(pca_fold, chm_fold, az_fold, img_size, rescale_pca=rescale_pca)
    train_loader = DataLoader(dataset, batch_size=1, num_workers=num_workers, pin_memory=True)
    model = models.SWaVModelStruct(patch_size, img_size, azm=azm, chm=chm, use_queue=use_queue, same_embed=same_embed, concat=concat, queue_chunks=queue_chunks, num_classes=num_classes, azm_concat=azm_concat, chm_concat=chm_concat, main_brightness=main_brightness, aug_brightness=aug_brightness)
    trainer = pl.Trainer(accelerator="gpu", max_epochs=max_epochs, callbacks=[checkpoint_callback]) #, accumulate_grad_batches=4
    trainer.fit(model, train_loader)

def do_sp_training(num_channels=10, num_classes=12, azm=False, chm=False, log_every=5, max_epochs=50, num_workers=4, extra_labels='', use_queue=False,  same_embed=False, concat=False, queue_chunks=1, azm_concat=False, chm_concat=False, main_brightness=False, aug_brightness=False):
    pca_fold = 'C:/Users/tonyt/Documents/Research/datasets/pca/niwo_masked_10'
    chm_fold = 'C:/Users/tonyt/Documents/Research/datasets/chm/niwo/'
    az_fold = 'C:/Users/tonyt/Documents/Research/datasets/solar_azimuth/niwo'
    sp_fold = 'C:/Users/tonyt/Documents/Research/datasets/superpixels/niwo/'

    checkpoint_callback = ModelCheckpoint(
        dirpath='ckpts', 
        filename=f'niwo_{num_channels}_channels_{num_classes}_classes_{extra_labels}'+'_{epoch}',
        every_n_epochs=log_every,
        save_on_train_epoch_end=True,
        save_top_k = -1
        )

    pl.seed_everything(42)

    dataset = MergedStructureDataset(pca_fold, chm_fold, az_fold, sp_fold, 4.015508459469479, 4.809300736115787, eval=False)
    train_loader = DataLoader(dataset, batch_size=1, num_workers=num_workers, pin_memory=True)
    model = models.SWaVModelSuperPixel(azm=azm, chm=chm, use_queue=use_queue, same_embed=same_embed, concat=concat, queue_chunks=queue_chunks, num_classes=num_classes, azm_concat=azm_concat, chm_concat=chm_concat, main_brightness=main_brightness, aug_brightness=aug_brightness)
    trainer = pl.Trainer(accelerator="gpu", max_epochs=max_epochs, callbacks=[checkpoint_callback])
    trainer.fit(model, train_loader)


def do_rendered_training(num_channels=10,
                        batch_size=2048,
                        num_classes=12, 
                        azm=False, 
                        chm=False, 
                        log_every=5, 
                        max_epochs=10, 
                        num_workers=4, 
                        extra_labels='', 
                        use_queue=False,  
                        queue_chunks=1, 
                        azm_concat=False, 
                        chm_concat=False, 
                        positions=False,
                        data_folder=None
                        ):
    data_folder = data_folder

    checkpoint_callback = ModelCheckpoint(
        dirpath='ckpts', 
        filename=f'niwo_{num_channels}_channels_{num_classes}_classes_{extra_labels}'+'_{epoch}',
        every_n_epochs=log_every,
        save_on_train_epoch_end=True,
        save_top_k = -1
        )

    pl.seed_everything(42)

    dataset = RenderedDataLoader(data_folder)
    train_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    model = models.SWaVModelSuperPixel(azm=azm, 
                                        chm=chm, 
                                        use_queue=use_queue, 
                                        queue_chunks=queue_chunks, 
                                        num_classes=num_classes, 
                                        azm_concat=azm_concat, 
                                        chm_concat=chm_concat,
                                        positions=positions
                                        )
    trainer = pl.Trainer(accelerator="gpu", max_epochs=max_epochs, callbacks=[checkpoint_callback])
    trainer.fit(model, train_loader)

def do_rendered_trainingresnet(num_channels=10,
                        batch_size=32,
                        num_classes=12, 
                        azm=False, 
                        chm=False, 
                        log_every=5, 
                        max_epochs=10, 
                        num_workers=4, 
                        extra_labels='', 
                        use_queue=False,  
                        queue_chunks=1, 
                        azm_concat=False, 
                        chm_concat=False, 
                        positions=False,
                        data_folder=None
                        ):
    data_folder = data_folder

    checkpoint_callback = ModelCheckpoint(
        dirpath='ckpts', 
        filename=f'niwo_{num_channels}_channels_{num_classes}_classes_resnet_{extra_labels}'+'_{epoch}',
        every_n_epochs=log_every,
        save_on_train_epoch_end=True,
        save_top_k = -1
        )

    pl.seed_everything(42)

    dataset = RenderedDataLoader(data_folder)
    train_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    model = models.SWaVModelSuperPixelResnet(azm=azm, 
                                        chm=chm, 
                                        use_queue=use_queue, 
                                        queue_chunks=queue_chunks, 
                                        num_classes=num_classes, 
                                        azm_concat=azm_concat, 
                                        chm_concat=chm_concat,
                                        positions=positions
                                        )
    trainer = pl.Trainer(accelerator="gpu", max_epochs=max_epochs, callbacks=[checkpoint_callback])
    trainer.fit(model, train_loader)

def refine(num_channels=10,
                        num_classes=12, 
                        azm=False, 
                        chm=False, 
                        log_every=25, 
                        max_epochs=200, 
                        num_workers=1, 
                        extra_labels='', 
                        use_queue=False,  
                        queue_chunks=1, 
                        azm_concat=False, 
                        chm_concat=False, 
                        positions=False,
                        data_folder=None,
                        valid_folder=None,
                        test_folder=None,
                        num_refine_classes = 5,
                        ckpt = None,
                        class_key = None,
                        class_weights = None,
                        freeze_backbone=True,
                        trained_backbone=True
                        ):
    data_folder = data_folder

    checkpoint_callback = ModelCheckpoint(
        dirpath='ckpts', 
        filename=f'niwo_{num_channels}_channels_{num_classes}_classes_refine_{extra_labels}'+'{ova:.2f}_{epoch}',
        #every_n_epochs=log_every,
        monitor='ova',
        save_on_train_epoch_end=True,
        mode='max',
        save_top_k = 5
        )

    pl.seed_everything(42)
    swav_config = {
        'num_classes': num_classes,
        'azm': azm,
        'chm': chm,
        'use_queue': use_queue,
        'queue_chunks': queue_chunks,
        'azm_concat': azm_concat,
        'chm_concat': chm_concat,
        'positions': positions,
        'ckpt': ckpt
    }

    train_dataset = RenderedDataLoader(data_folder)
    train_loader = DataLoader(train_dataset, batch_size=32, num_workers=num_workers)

    valid_dataset = RenderedDataLoader(valid_folder)
    valid_loader = DataLoader(valid_dataset, batch_size=1, num_workers=num_workers)

    test_dataset = RenderedDataLoader(test_folder)
    test_loader = DataLoader(test_dataset, batch_size=1, num_workers=num_workers)

   
    model = models.SWaVModelRefine(swav_config, num_refine_classes, class_key=class_key, class_weights=class_weights, freeze_backbone=freeze_backbone, trained_backbone=trained_backbone)
    trainer = pl.Trainer(accelerator="gpu", max_epochs=max_epochs, callbacks=[checkpoint_callback])
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=valid_loader)
    trainer.test(ckpt_path="best", dataloaders=test_loader)



if __name__ == "__main__":

    # refine(data_folder='C:/Users/tonyt/Documents/Research/datasets/tensors/niwo_2020_10_pca_ndvi_masked/label_training',
    #         valid_folder= 'C:/Users/tonyt/Documents/Research/datasets/tensors/niwo_2020_10_pca_ndvi_masked/label_valid',
    #         test_folder = 'C:/Users/tonyt/Documents/Research/datasets/tensors/niwo_2020_10_pca_ndvi_masked/label_test',
    #         azm=True,
    #         chm_concat=True,
    #         num_classes=256,
    #         extra_labels='lr_5e4_400_untrained_pca_pca',
    #         class_key= {0: 'PIEN', 1: 'ABLAL', 2: 'PIFL2', 3: 'PICOL', 4: 'SALIX'},
    #         class_weights= [0.47228916, 0.60775194, 3.26666667, 1.225, 8.71111111],
    #         positions=False,
    #         freeze_backbone=False,
    #         trained_backbone=False,
    #         ckpt='ckpts/niwo_10_channels_256_classes_azm_add_chm_concat_pre_rendered_per_pixel_epoch=9.ckpt')





    # do_rendered_trainingresnet(num_workers=8, num_classes=256, azm=True, chm_concat=True, extra_labels='azm_add_chm_concat_pre_rendered_per_pixel',
    #                             data_folder='C:/Users/tonyt/Documents/Research/datasets/tensors/niwo_2020_10_pca_ndvi_masked/super_pixel_patches/raw_training/')
    

    # configs = [
    #     {'num_channels': 10,
    #      'num_classes': 12,
    #      'azm': True,
    #      'chm': True,
    #      'patch_size': 4,
    #      'log_every': 10,
    #      'max_epochs': 50,
    #      'num_workers': 4,
    #      'img_size': 40,
    #      'extra_labels': 'chm_and_azm'
    #      },
    #       {'num_channels': 10,
    #      'num_classes': 12,
    #      'azm': False, ##FUCKED THIS ONE UP - REDO
    #      'chm': True,
    #      'patch_size': 4,
    #      'log_every': 10,
    #      'max_epochs': 50,
    #      'num_workers': 4,
    #      'img_size': 40,
    #      'extra_labels': 'azm_only'
    #      },
    #       {'num_channels': 10,
    #      'num_classes': 12,
    #      'azm': False,
    #      'chm': True,
    #      'patch_size': 4,
    #      'log_every': 10,
    #      'max_epochs': 50,
    #      'num_workers': 4,
    #      'img_size': 40,
    #      'extra_labels': 'chm_only'
    #      },
    #      {'num_channels': 10,
    #      'num_classes': 12,
    #      'azm': False,
    #      'chm': False,
    #      'patch_size': 4,
    #      'log_every': 10,
    #      'max_epochs': 50,
    #      'num_workers': 4,
    #      'img_size': 40,
    #      'extra_labels': 'no_structure'
    #      },
    #      {'num_channels': 10,
    #      'num_classes': 12,
    #      'azm': True,
    #      'chm': True,
    #      'patch_size': 4,
    #      'log_every': 10,
    #      'max_epochs': 50,
    #      'num_workers': 4,
    #      'img_size': 40,
    #      'use_queue': True,
    #      'extra_labels': 'all_struct_queue'
    #      },
    #      {'num_channels': 10,
    #      'num_classes': 12,
    #      'azm': False,
    #      'chm': False,
    #      'patch_size': 4,
    #      'log_every': 10,
    #      'max_epochs': 50,
    #      'num_workers': 4,
    #      'img_size': 40,
    #      'use_queue': True,
    #      'extra_labels': 'no_struct_queue'
    #      }
    # ]

    # configs = [
    #       {'num_channels': 10,
    #      'num_classes': 12,
    #      'azm': True,
    #      'chm': False,
    #      'patch_size': 4,
    #      'log_every': 10,
    #      'max_epochs': 50,
    #      'num_workers': 4,
    #      'img_size': 40,
    #      'extra_labels': 'azm_only'
    #      },
    #     {'num_channels': 10,
    #      'num_classes': 12,
    #      'azm': True,
    #      'chm': False,
    #      'patch_size': 4,
    #      'log_every': 10,
    #      'max_epochs': 50,
    #      'num_workers': 4,
    #      'img_size': 40,
    #      'use_queue': True,
    #      'extra_labels': 'azm_only_queue'
    #      },
    #      {'num_channels': 10,
    #      'num_classes': 12,
    #      'azm': False,
    #      'chm': True,
    #      'patch_size': 4,
    #      'log_every': 10,
    #      'max_epochs': 50,
    #      'num_workers': 4,
    #      'img_size': 40,
    #      'use_queue': True,
    #      'extra_labels': 'chm_only_queue'
    #      },
    #      {'num_channels': 10,
    #      'num_classes': 12,
    #      'azm': True,
    #      'chm': True,
    #      'patch_size': 4,
    #      'log_every': 10,
    #      'max_epochs': 50,
    #      'num_workers': 4,
    #      'img_size': 40,
    #      'use_queue': False,
    #      'same_embed': True,
    #      'extra_labels': 'struct_same_embed'
    #      },
    #      {'num_channels': 10,
    #      'num_classes': 60,
    #      'azm': True,
    #      'chm': True,
    #      'patch_size': 4,
    #      'log_every': 10,
    #      'max_epochs': 50,
    #      'num_workers': 4,
    #      'img_size': 40,
    #      'use_queue': False,
    #      'same_embed': False,
    #      'extra_labels': 'struct_more_classes'
    #      },
    #      {'num_channels': 10,
    #      'num_classes': 60,
    #      'azm': True,
    #      'chm': True,
    #      'patch_size': 4,
    #      'log_every': 10,
    #      'max_epochs': 50,
    #      'num_workers': 4,
    #      'img_size': 40,
    #      'use_queue': True,
    #      'same_embed': False,
    #      'extra_labels': 'struct_more_classes_queue'
    #      },
    #      {'num_channels': 10,
    #      'num_classes': 12,
    #      'azm': True,
    #      'chm': True,
    #      'patch_size': 4,
    #      'log_every': 10,
    #      'max_epochs': 50,
    #      'num_workers': 4,
    #      'img_size': 40,
    #      'use_queue': False,
    #      'same_embed': False,
    #      'concat': True,
    #      'extra_labels': 'struct_concat'
    #      },
    #      {'num_channels': 10,
    #      'num_classes': 12,
    #      'azm': True,
    #      'chm': True,
    #      'patch_size': 4,
    #      'log_every': 10,
    #      'max_epochs': 50,
    #      'num_workers': 4,
    #      'img_size': 40,
    #      'use_queue': True,
    #      'same_embed': False,
    #      'concat': True,
    #      'extra_labels': 'struct_concat_queue'
    #      },
    #      {'num_channels': 10,
    #      'num_classes': 12,
    #      'azm': True,
    #      'chm': True,
    #      'patch_size': 4,
    #      'log_every': 10,
    #      'max_epochs': 50,
    #      'num_workers': 4,
    #      'img_size': 40,
    #      'use_queue': True,
    #      'same_embed': False,
    #      'concat': False,
    #      'queue_chunks': 5,
    #      'extra_labels': 'struct_queue_5_chunks'
    #      },
    #      {'num_channels': 10,
    #      'num_classes': 12,
    #      'azm': False,
    #      'chm': False,
    #      'patch_size': 4,
    #      'log_every': 10,
    #      'max_epochs': 50,
    #      'num_workers': 4,
    #      'img_size': 40,
    #      'use_queue': True,
    #      'same_embed': False,
    #      'concat': False,
    #      'queue_chunks': 5,
    #      'extra_labels': 'no_struct_queue_5_chunks'
    #      }
    #     ]

    # configs = [
    #     {'num_channels': 10,
    #      'num_classes': 12,
    #      'azm': True,
    #      'chm': True,
    #      'patch_size': 4,
    #      'log_every': 10,
    #      'max_epochs': 50,
    #      'num_workers': 4,
    #      'img_size': 40,
    #      'use_queue': True,
    #      'same_embed': False,
    #      'azm_concat': True,
    #      'chm_concat': True,
    #      'queue_chunks': 5,
    #      'extra_labels': 'all_struct_concat_queue_5_chunks'
    #      },
    #      {'num_channels': 10,
    #      'num_classes': 12,
    #      'azm': True,
    #      'chm': True,
    #      'patch_size': 4,
    #      'log_every': 10,
    #      'max_epochs': 50,
    #      'num_workers': 4,
    #      'img_size': 40,
    #      'use_queue': False,
    #      'same_embed': False,
    #      'azm_concat': False,
    #      'chm_concat': True,
    #      'queue_chunks': 5,
    #      'extra_labels': 'azm_add_chm_concat_no_queue'
    #      },
    #       {'num_channels': 10,
    #      'num_classes': 12,
    #      'azm': True,
    #      'chm': True,
    #      'patch_size': 4,
    #      'log_every': 10,
    #      'max_epochs': 50,
    #      'num_workers': 4,
    #      'img_size': 40,
    #      'use_queue': False,
    #      'same_embed': False,
    #      'azm_concat': True,
    #      'chm_concat': False,
    #      'queue_chunks': 5,
    #      'extra_labels': 'chm_add_azm_concat_no_queue'
    #      },
    #       {'num_channels': 10,
    #      'num_classes': 12,
    #      'azm': False,
    #      'chm': True,
    #      'patch_size': 4,
    #      'log_every': 10,
    #      'max_epochs': 50,
    #      'num_workers': 4,
    #      'img_size': 40,
    #      'use_queue': False,
    #      'same_embed': False,
    #      'azm_concat': False,
    #      'chm_concat': True,
    #      'queue_chunks': 5,
    #      'extra_labels': 'chm_concat_only_no_queue'
    #      },
    #       {'num_channels': 10,
    #      'num_classes': 12,
    #      'azm': True,
    #      'chm': False,
    #      'patch_size': 4,
    #      'log_every': 10,
    #      'max_epochs': 50,
    #      'num_workers': 4,
    #      'img_size': 40,
    #      'use_queue': False,
    #      'same_embed': False,
    #      'azm_concat': True,
    #      'chm_concat': False,
    #      'queue_chunks': 5,
    #      'extra_labels': 'azm_concat_only_no_queue'
    #      },
    #       {'num_channels': 10,
    #      'num_classes': 60,
    #      'azm': True,
    #      'chm': True,
    #      'patch_size': 4,
    #      'log_every': 10,
    #      'max_epochs': 50,
    #      'num_workers': 4,
    #      'img_size': 40,
    #      'use_queue': False,
    #      'same_embed': False,
    #      'azm_concat': True,
    #      'chm_concat': True,
    #      'queue_chunks': 5,
    #      'extra_labels': 'all_concat_60_classes_no_queue'
    #      },
    #       {'num_channels': 10,
    #      'num_classes': 256,
    #      'azm': True,
    #      'chm': True,
    #      'patch_size': 4,
    #      'log_every': 10,
    #      'max_epochs': 50,
    #      'num_workers': 4,
    #      'img_size': 40,
    #      'use_queue': False,
    #      'same_embed': False,
    #      'azm_concat': True,
    #      'chm_concat': True,
    #      'queue_chunks': 5,
    #      'extra_labels': 'all_concat_256_classes_no_queue'
    #      },
    # ]

    # configs = [
    #     {'num_channels': 10,
    #      'num_classes': 12,
    #      'azm': True,
    #      'chm': True,
    #      'patch_size': 4,
    #      'log_every': 10,
    #      'max_epochs': 50,
    #      'num_workers': 4,
    #      'img_size': 40,
    #      'use_queue': False,
    #      'same_embed': False,
    #      'azm_concat': True,
    #      'chm_concat': True,
    #      'queue_chunks': 1,
    #      'main_brightness': True,
    #      'aug_brightness': True,
    #      'rescale_pca': True,
    #      'extra_labels': 'all_struct_concat_both_brightness'
    #      },
    #      {'num_channels': 10,
    #      'num_classes': 12,
    #      'azm': True,
    #      'chm': True,
    #      'patch_size': 4,
    #      'log_every': 10,
    #      'max_epochs': 50,
    #      'num_workers': 4,
    #      'img_size': 40,
    #      'use_queue': False,
    #      'same_embed': False,
    #      'azm_concat': True,
    #      'chm_concat': True,
    #      'queue_chunks': 1,
    #      'main_brightness': False,
    #      'aug_brightness': True,
    #      'rescale_pca': True,
    #      'extra_labels': 'all_struct_concat_only_aug_brightness'
    #      },
    #      {'num_channels': 10,
    #      'num_classes': 12,
    #      'azm': True,
    #      'chm': True,
    #      'patch_size': 4,
    #      'log_every': 10,
    #      'max_epochs': 50,
    #      'num_workers': 4,
    #      'img_size': 40,
    #      'use_queue': False,
    #      'same_embed': False,
    #      'azm_concat': True,
    #      'chm_concat': True,
    #      'queue_chunks': 1,
    #      'main_brightness': True,
    #      'aug_brightness': False,
    #      'rescale_pca': True,
    #      'extra_labels': 'all_struct_concat_only_main_brightness'
    #      },
    #      {'num_channels': 10,
    #      'num_classes': 12,
    #      'azm': True,
    #      'chm': True,
    #      'patch_size': 4,
    #      'log_every': 10,
    #      'max_epochs': 50,
    #      'num_workers': 4,
    #      'img_size': 40,
    #      'use_queue': False,
    #      'same_embed': False,
    #      'azm_concat': True,
    #      'chm_concat': True,
    #      'queue_chunks': 1,
    #      'main_brightness': False,
    #      'aug_brightness': False,
    #      'rescale_pca': True,
    #      'extra_labels': 'concat_only_rescale'
    #      }
    #  ]

    # configs = [
    #     {'num_channels': 10,
    #      'num_classes': 30,
    #      'azm': True,
    #      'chm': True,
    #      'patch_size': 4,
    #      'log_every': 10,
    #      'max_epochs': 50,
    #      'num_workers': 4,
    #      'img_size': 40,
    #      'use_queue': False,
    #      'same_embed': False,
    #      'azm_concat': True,
    #      'chm_concat': False,
    #      'queue_chunks': 1,
    #      'main_brightness': True,
    #      'aug_brightness': False,
    #      'rescale_pca': True,
    #      'extra_labels': 'concat_azm_add_chm_main_brightness_30_classes'
    #      },
    # ]

    # configs = [
    #     {'num_channels': 10,
    #      'num_classes': 30,
    #      'azm': True,
    #      'chm': True,
    #      'patch_size': 4,
    #      'log_every': 10,
    #      'max_epochs': 50,
    #      'num_workers': 4,
    #      'img_size': 40,
    #      'use_queue': True,
    #      'same_embed': False,
    #      'azm_concat': True,
    #      'chm_concat': False,
    #      'queue_chunks': 20,
    #      'main_brightness': False,
    #      'aug_brightness': False,
    #      'rescale_pca': False,
    #      'extra_labels': 'azm_concat_chm_add_queue_chunks_30'
    #      },
    #      {'num_channels': 10,
    #      'num_classes': 12,
    #      'azm': True,
    #      'chm': True,
    #      'patch_size': 4,
    #      'log_every': 10,
    #      'max_epochs': 50,
    #      'num_workers': 4,
    #      'img_size': 40,
    #      'use_queue': True,
    #      'same_embed': False,
    #      'azm_concat': True,
    #      'chm_concat': False,
    #      'queue_chunks': 20,
    #      'main_brightness': False,
    #      'aug_brightness': False,
    #      'rescale_pca': False,
    #      'extra_labels': 'azm_concat_chm_add_queue_chunks_30'
    #      },
    # ]

    configs = [
        # {'num_channels': 10,
        #  'num_classes': 256,
        #  'azm': True,
        #  'chm': True,
        #  'patch_size': 4,
        #  'log_every': 10,
        #  'max_epochs': 50,
        #  'num_workers': 4,
        #  'img_size': 40,
        #  'use_queue': True,
        #  'same_embed': False,
        #  'azm_concat': True,
        #  'chm_concat': False,
        #  'queue_chunks': 20,
        #  'main_brightness': False,
        #  'aug_brightness': False,
        #  'rescale_pca': False,
        #  'extra_labels': 'azm_concat_chm_add_queue_chunks_20'
        #  },
        #  {'num_channels': 10,
        #  'num_classes': 12,
        #  'azm': True,
        #  'chm': True,
        #  'patch_size': 4,
        #  'log_every': 10,
        #  'max_epochs': 50,
        #  'num_workers': 4,
        #  'img_size': 40,
        #  'use_queue': True,
        #  'same_embed': False,
        #  'azm_concat': True,
        #  'chm_concat': False,
        #  'queue_chunks': 10,
        #  'main_brightness': False,
        #  'aug_brightness': False,
        #  'rescale_pca': False,
        #  'extra_labels': 'azm_concat_chm_add_queue_chunks_10'
        #  },
        #  {'num_channels': 10,
        #  'num_classes': 20,
        #  'azm': True,
        #  'chm': True,
        #  'patch_size': 4,
        #  'log_every': 10,
        #  'max_epochs': 50,
        #  'num_workers': 4,
        #  'img_size': 40,
        #  'use_queue': True,
        #  'same_embed': False,
        #  'azm_concat': True,
        #  'chm_concat': False,
        #  'queue_chunks': 20,
        #  'main_brightness': False,
        #  'aug_brightness': False,
        #  'rescale_pca': False,
        #  'extra_labels': 'azm_concat_chm_add_queue_chunks_10'
        #  },
        #  {'num_channels': 10,
        #  'num_classes': 12,
        #  'azm': True,
        #  'chm': True,
        #  'patch_size': 4,
        #  'log_every': 10,
        #  'max_epochs': 50,
        #  'num_workers': 4,
        #  'img_size': 40,
        #  'use_queue': False,
        #  'same_embed': False,
        #  'azm_concat': True,
        #  'chm_concat': False,
        #  'queue_chunks': 20,
        #  'main_brightness': False,
        #  'aug_brightness': False,
        #  'rescale_pca': False,
        #  'extra_labels': 'azm_concat_chm_add'
        #  },
        #  {'num_channels': 10,
        #  'num_classes': 12,
        #  'azm': False,
        #  'chm': False,
        #  'patch_size': 4,
        #  'log_every': 10,
        #  'max_epochs': 50,
        #  'num_workers': 4,
        #  'img_size': 40,
        #  'use_queue': False,
        #  'same_embed': False,
        #  'azm_concat': True,
        #  'chm_concat': False,
        #  'queue_chunks': 20,
        #  'main_brightness': False,
        #  'aug_brightness': False,
        #  'rescale_pca': False,
        #  'extra_labels': 'no_struct'
        #  },
         {'num_channels': 10,
         'num_classes': 12,
         'azm': False,
         'chm': False,
         'patch_size': 4,
         'log_every': 10,
         'max_epochs': 50,
         'num_workers': 4,
         'img_size': 40,
         'use_queue': True,
         'same_embed': False,
         'azm_concat': True,
         'chm_concat': False,
         'queue_chunks': 10,
         'main_brightness': False,
         'aug_brightness': False,
         'rescale_pca': False,
         'extra_labels': 'no_struct_queue_10_chunks'
         },
         {'num_channels': 10,
         'num_classes': 12,
         'azm': True,
         'chm': True,
         'patch_size': 4,
         'log_every': 10,
         'max_epochs': 50,
         'num_workers': 4,
         'img_size': 40,
         'use_queue': True,
         'same_embed': False,
         'azm_concat': True,
         'chm_concat': False,
         'queue_chunks': 10,
         'main_brightness': False,
         'aug_brightness': False,
         'rescale_pca': False,
         'extra_labels': 'azm_concat_chm_add_queue_10_chunks'
         },
    ]
    #TODO: No struct rescale
    
    # for config in configs:
    #     do_training(**config)

    