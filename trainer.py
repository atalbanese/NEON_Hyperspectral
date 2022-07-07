from dataloaders import RenderedDataLoader
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import models
from pytorch_lightning.callbacks import ModelCheckpoint, StochasticWeightAveraging
import inference
import warnings




def do_rendered_training(num_channels=31,
                        batch_size=2048,
                        num_classes=12, 
                        azm=False, 
                        chm=False, 
                        log_every=20, 
                        max_epochs=200, 
                        num_workers=4, 
                        extra_labels='', 
                        use_queue=False,  
                        queue_chunks=5, 
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



def refine(chm_mean=None, chm_std =None, num_channels=31,
                        num_classes=12, 
                        azm=False, 
                        chm=False, 
                        log_every=50, 
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
                        num_refine_classes = 4,
                        ckpt = None,
                        class_key = None,
                        class_weights = None,
                        freeze_backbone=True,
                        trained_backbone=True,
                        height_threshold=5
                        ):
    data_folder = data_folder

    checkpoint_callback = ModelCheckpoint(
        dirpath='ckpts/sp_training', 
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

   
    model = models.SWaVModelRefine(swav_config, num_refine_classes, class_key, chm_mean, chm_std, class_weights=class_weights, freeze_backbone=freeze_backbone, trained_backbone=trained_backbone, height_threshold=height_threshold)
    trainer = pl.Trainer(accelerator="gpu", max_epochs=max_epochs, callbacks=[checkpoint_callback, StochasticWeightAveraging(50)])
    #trainer.tune(model, train_dataloaders=train_loader, val_dataloaders=valid_loader)
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=valid_loader)
    trainer.test(ckpt_path='best', dataloaders=test_loader)

def unified_training(class_key,
                    chm_mean,
                    chm_std,
                    lr,
                    height_threshold,
                    class_weights,
                    trained_backbone,
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
                    extra_labels):

    pl.seed_everything(42)

    refine_callback = ModelCheckpoint(
        dirpath='ckpts/unified_training', 
        filename=f'niwo_refine_{extra_labels}'+'{ova:.2f}_{epoch}',
        #every_n_epochs=log_every,
        monitor='ova',
        save_on_train_epoch_end=True,
        mode='max',
        save_top_k = 5
        )

    pre_train_callback = ModelCheckpoint(
        dirpath='ckpts/unified_training', 
        filename=f'niwo_pre_train_{extra_labels}'+'_{epoch}',
        every_n_epochs=pre_training_epochs,
        save_on_train_epoch_end=True,
        save_top_k = -1
        )
    
    pre_train_ckpt = f'ckpts/unified_training/niwo_pre_train_{extra_labels}_epoch={pre_training_epochs-1}.ckpt'

    pre_model = models.SwaVModelUnified(class_key,
                                    chm_mean,
                                    chm_std,
                                    lr,
                                    height_threshold,
                                    class_weights,
                                    trained_backbone,
                                    features_dict,
                                    num_intermediate_classes,
                                    pre_training=True)

    pre_train_data = RenderedDataLoader(pre_train_folder)
    pre_train_loader = DataLoader(pre_train_data, batch_size=pre_train_batch_size, num_workers=pre_train_workers, pin_memory=True)

    train_dataset = RenderedDataLoader(train_folder)
    train_loader = DataLoader(train_dataset, batch_size=refine_batch_size, num_workers=refine_workers)

    valid_dataset = RenderedDataLoader(valid_folder)
    valid_loader = DataLoader(valid_dataset, batch_size=1, num_workers=refine_workers)

    test_dataset = RenderedDataLoader(test_folder)
    test_loader = DataLoader(test_dataset, batch_size=1, num_workers=refine_workers)

    pre_trainer = pl.Trainer(accelerator="gpu", max_epochs=pre_training_epochs, callbacks=[pre_train_callback])
    refiner = pl.Trainer(accelerator="gpu", max_epochs=refine_epochs, callbacks=[refine_callback])

    pre_trainer.fit(pre_model, pre_train_loader)

    refine_model = models.SwaVModelUnified.load_from_checkpoint(pre_train_ckpt, pre_training=False)
    refiner.fit(refine_model, train_dataloaders=train_loader, val_dataloaders=valid_loader)
    refiner.test(ckpt_path='best', dataloaders=test_loader)
    


    




if __name__ == "__main__":

    #TODO:
    #HPARAM TUNING
    #TRY WITHOUT CHM FILTER
    #TRY FEWER INPUTS

    unified_training(class_key={0: 'PIEN', 1: 'ABLAL', 2: 'PICOL', 3: 'PIFL2'},
                    chm_mean=4.015508459469479,
                    chm_std=4.809300736115787,
                    lr=5e-4,
                    height_threshold=5,
                    class_weights=[0.6434426229508197, 0.7405660377358491, 1.353448275862069, 2.8035714285714284],
                    trained_backbone=True,
                    features_dict={
                        'pca' : 10,
                        'raw_bands': 8,
                        'shadow': 1, 
                        'chm': 1,
                        'azm': 1 
                    },
                    num_intermediate_classes = 256,
                    pre_train_folder = 'C:/Users/tonyt/Documents/Research/datasets/tensors/niwo_2020_pca_ica_shadow_extra/raw_training/',
                    train_folder = 'C:/Users/tonyt/Documents/Research/datasets/tensors/niwo_2020_pca_ica_shadow_extra/label_training',
                    valid_folder = 'C:/Users/tonyt/Documents/Research/datasets/tensors/niwo_2020_pca_ica_shadow_extra/label_valid',
                    test_folder  = 'C:/Users/tonyt/Documents/Research/datasets/tensors/niwo_2020_pca_ica_shadow_extra/label_test',
                    pre_training_epochs = 50,
                    refine_epochs = 400,
                    pre_train_batch_size = 2048,
                    refine_batch_size = 32,
                    pre_train_workers = 8,
                    refine_workers = 1,
                    extra_labels= 'pca_raw_shadow_chm_azm')

    unified_training(class_key={0: 'PIEN', 1: 'ABLAL', 2: 'PICOL', 3: 'PIFL2'},
                    chm_mean=4.015508459469479,
                    chm_std=4.809300736115787,
                    lr=5e-4,
                    height_threshold=5,
                    class_weights=[0.6434426229508197, 0.7405660377358491, 1.353448275862069, 2.8035714285714284],
                    trained_backbone=True,
                    features_dict={
                        'ica': 10,
                        'raw_bands': 8,
                        'shadow': 1, 
                        'chm': 1,
                        'azm': 1 
                    },
                    num_intermediate_classes = 256,
                    pre_train_folder = 'C:/Users/tonyt/Documents/Research/datasets/tensors/niwo_2020_pca_ica_shadow_extra/raw_training/',
                    train_folder = 'C:/Users/tonyt/Documents/Research/datasets/tensors/niwo_2020_pca_ica_shadow_extra/label_training',
                    valid_folder = 'C:/Users/tonyt/Documents/Research/datasets/tensors/niwo_2020_pca_ica_shadow_extra/label_valid',
                    test_folder  = 'C:/Users/tonyt/Documents/Research/datasets/tensors/niwo_2020_pca_ica_shadow_extra/label_test',
                    pre_training_epochs = 50,
                    refine_epochs = 400,
                    pre_train_batch_size = 2048,
                    refine_batch_size = 32,
                    pre_train_workers = 8,
                    refine_workers = 1,
                    extra_labels= 'ica_raw_shadow_chm_azm')

    unified_training(class_key={0: 'PIEN', 1: 'ABLAL', 2: 'PICOL', 3: 'PIFL2'},
                    chm_mean=4.015508459469479,
                    chm_std=4.809300736115787,
                    lr=5e-4,
                    height_threshold=5,
                    class_weights=[0.6434426229508197, 0.7405660377358491, 1.353448275862069, 2.8035714285714284],
                    trained_backbone=True,
                    features_dict={
                        'pca': 10,
                        'ica': 10,
                        'shadow': 1, 
                        'chm': 1,
                        'azm': 1 
                    },
                    num_intermediate_classes = 256,
                    pre_train_folder = 'C:/Users/tonyt/Documents/Research/datasets/tensors/niwo_2020_pca_ica_shadow_extra/raw_training/',
                    train_folder = 'C:/Users/tonyt/Documents/Research/datasets/tensors/niwo_2020_pca_ica_shadow_extra/label_training',
                    valid_folder = 'C:/Users/tonyt/Documents/Research/datasets/tensors/niwo_2020_pca_ica_shadow_extra/label_valid',
                    test_folder  = 'C:/Users/tonyt/Documents/Research/datasets/tensors/niwo_2020_pca_ica_shadow_extra/label_test',
                    pre_training_epochs = 50,
                    refine_epochs = 400,
                    pre_train_batch_size = 2048,
                    refine_batch_size = 32,
                    pre_train_workers = 8,
                    refine_workers = 1,
                    extra_labels= 'pca_ica_shadow_chm_azm')
    
    unified_training(class_key={0: 'PIEN', 1: 'ABLAL', 2: 'PICOL', 3: 'PIFL2'},
                    chm_mean=4.015508459469479,
                    chm_std=4.809300736115787,
                    lr=5e-4,
                    height_threshold=5,
                    class_weights=[0.6434426229508197, 0.7405660377358491, 1.353448275862069, 2.8035714285714284],
                    trained_backbone=True,
                    features_dict={
                        'raw_bands': 8,
                        'shadow': 1, 
                        'chm': 1,
                        'azm': 1 
                    },
                    num_intermediate_classes = 256,
                    pre_train_folder = 'C:/Users/tonyt/Documents/Research/datasets/tensors/niwo_2020_pca_ica_shadow_extra/raw_training/',
                    train_folder = 'C:/Users/tonyt/Documents/Research/datasets/tensors/niwo_2020_pca_ica_shadow_extra/label_training',
                    valid_folder = 'C:/Users/tonyt/Documents/Research/datasets/tensors/niwo_2020_pca_ica_shadow_extra/label_valid',
                    test_folder  = 'C:/Users/tonyt/Documents/Research/datasets/tensors/niwo_2020_pca_ica_shadow_extra/label_test',
                    pre_training_epochs = 50,
                    refine_epochs = 400,
                    pre_train_batch_size = 2048,
                    refine_batch_size = 32,
                    pre_train_workers = 8,
                    refine_workers = 1,
                    extra_labels= 'raw_shadow_chm_azm')

    unified_training(class_key={0: 'PIEN', 1: 'ABLAL', 2: 'PICOL', 3: 'PIFL2'},
                    chm_mean=4.015508459469479,
                    chm_std=4.809300736115787,
                    lr=5e-4,
                    height_threshold=5,
                    class_weights=[0.6434426229508197, 0.7405660377358491, 1.353448275862069, 2.8035714285714284],
                    trained_backbone=True,
                    features_dict={
                        'pca' : 10,
                        'ica': 10,
                        'raw_bands': 8
                    },
                    num_intermediate_classes = 256,
                    pre_train_folder = 'C:/Users/tonyt/Documents/Research/datasets/tensors/niwo_2020_pca_ica_shadow_extra/raw_training/',
                    train_folder = 'C:/Users/tonyt/Documents/Research/datasets/tensors/niwo_2020_pca_ica_shadow_extra/label_training',
                    valid_folder = 'C:/Users/tonyt/Documents/Research/datasets/tensors/niwo_2020_pca_ica_shadow_extra/label_valid',
                    test_folder  = 'C:/Users/tonyt/Documents/Research/datasets/tensors/niwo_2020_pca_ica_shadow_extra/label_test',
                    pre_training_epochs = 50,
                    refine_epochs = 400,
                    pre_train_batch_size = 2048,
                    refine_batch_size = 32,
                    pre_train_workers = 8,
                    refine_workers = 1,
                    extra_labels= 'pca_ica_raw')

    unified_training(class_key={0: 'PIEN', 1: 'ABLAL', 2: 'PICOL', 3: 'PIFL2'},
                    chm_mean=4.015508459469479,
                    chm_std=4.809300736115787,
                    lr=5e-4,
                    height_threshold=5,
                    class_weights=[0.6434426229508197, 0.7405660377358491, 1.353448275862069, 2.8035714285714284],
                    trained_backbone=True,
                    features_dict={
                        'pca' : 3,
                        'ica': 3,
                        'raw_bands': 8,
                        'shadow': 1, 
                        'chm': 1,
                        'azm': 1 
                    },
                    num_intermediate_classes = 256,
                    pre_train_folder = 'C:/Users/tonyt/Documents/Research/datasets/tensors/niwo_2020_pca_ica_shadow_extra/raw_training/',
                    train_folder = 'C:/Users/tonyt/Documents/Research/datasets/tensors/niwo_2020_pca_ica_shadow_extra/label_training',
                    valid_folder = 'C:/Users/tonyt/Documents/Research/datasets/tensors/niwo_2020_pca_ica_shadow_extra/label_valid',
                    test_folder  = 'C:/Users/tonyt/Documents/Research/datasets/tensors/niwo_2020_pca_ica_shadow_extra/label_test',
                    pre_training_epochs = 50,
                    refine_epochs = 400,
                    pre_train_batch_size = 2048,
                    refine_batch_size = 32,
                    pre_train_workers = 8,
                    refine_workers = 1,
                    extra_labels= 'pca_3_ica_3_raw_shadow_chm_azm')

    unified_training(class_key={0: 'PIEN', 1: 'ABLAL', 2: 'PICOL', 3: 'PIFL2'},
                    chm_mean=4.015508459469479,
                    chm_std=4.809300736115787,
                    lr=5e-4,
                    height_threshold=5,
                    class_weights=[0.6434426229508197, 0.7405660377358491, 1.353448275862069, 2.8035714285714284],
                    trained_backbone=True,
                    features_dict={
                        'pca' : 3,
                        'ica': 3,
                        'raw_bands': 8,
                    },
                    num_intermediate_classes = 256,
                    pre_train_folder = 'C:/Users/tonyt/Documents/Research/datasets/tensors/niwo_2020_pca_ica_shadow_extra/raw_training/',
                    train_folder = 'C:/Users/tonyt/Documents/Research/datasets/tensors/niwo_2020_pca_ica_shadow_extra/label_training',
                    valid_folder = 'C:/Users/tonyt/Documents/Research/datasets/tensors/niwo_2020_pca_ica_shadow_extra/label_valid',
                    test_folder  = 'C:/Users/tonyt/Documents/Research/datasets/tensors/niwo_2020_pca_ica_shadow_extra/label_test',
                    pre_training_epochs = 50,
                    refine_epochs = 400,
                    pre_train_batch_size = 2048,
                    refine_batch_size = 32,
                    pre_train_workers = 8,
                    refine_workers = 1,
                    extra_labels= 'pca_3_ica_3_raw')

    # refine(data_folder='C:/Users/tonyt/Documents/Research/datasets/tensors/niwo_2020_pca_ica_shadow_extra/label_training',
    #         valid_folder= 'C:/Users/tonyt/Documents/Research/datasets/tensors/niwo_2020_pca_ica_shadow_extra/label_valid',
    #         test_folder = 'C:/Users/tonyt/Documents/Research/datasets/tensors/niwo_2020_pca_ica_shadow_extra/label_test',
    #         num_classes=256,
    #         extra_labels='trained_backbone_superpixel_training_data_lr_5e4_NO_BN_SWA',
    #         class_key= {0: 'PIEN', 1: 'ABLAL', 2: 'PICOL', 3: 'PIFL2'},
    #         class_weights= [0.6434426229508197, 0.7405660377358491, 1.353448275862069, 2.8035714285714284],
    #         chm_mean = 4.015508459469479,
    #         chm_std = 4.809300736115787,
    #         positions=False,
    #         freeze_backbone=False,
    #         trained_backbone=True,
    #         use_queue=False,
    #         ckpt='ckpts/niwo_31_channels_256_classes_pca_ica_shadow_extra_400_epochs_epoch=199.ckpt')
    




    # do_rendered_training(num_workers=8, num_classes=256, extra_labels='pca_ica_shadow_extra_400_epochs', use_queue=False,
    #                             data_folder='C:/Users/tonyt/Documents/Research/datasets/tensors/niwo_2020_pca_ica_shadow_extra/raw_training/')

    # do_rendered_training(num_workers=8, num_classes=256, extra_labels='pca_ica_shadow_extra_queue', use_queue=True,
    #                             data_folder='C:/Users/tonyt/Documents/Research/datasets/tensors/niwo_2020_pca_ica_shadow_extra/raw_training/')
    
    

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

    