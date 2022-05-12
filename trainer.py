from dataloaders import HyperDataset, PreProcDataset, MaskedDataset, MaskedVitDataset, MaskedDenseVitDataset, DeepClusterDataset, StructureDataset
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import models
from pytorch_lightning.callbacks import ModelCheckpoint, StochasticWeightAveraging
import inference
import warnings


if __name__ == "__main__":
    #warnings.filterwarnings("ignore")
    pca_fold = 'C:/Users/tonyt/Documents/Research/datasets/pca/harv_2022_10_channels'
    chm_fold = 'C:/Users/tonyt/Documents/Research/datasets/chm/harv_2019/NEON_struct-ecosystem/NEON.D01.HARV.DP3.30015.001.2019-08.basic.20220511T165943Z.RELEASE-2022'
    az_fold = 'C:/Users/tonyt/Documents/Research/datasets/solar_azimuth/harv_2022'

    
    #h5_fold = "/data/shared/src/aalbanese/datasets/hs/NEON_refl-surf-dir-ortho-mosaic/NEON.D16.WREF.DP3.30006.001.2021-07.basic.20220330T192306Z.PROVISIONAL"
    h5_fold = "/data/shared/src/aalbanese/datasets/hs/NEON_refl-surf-dir-ortho-mosaic/NEON.D01.HARV.DP3.30006.001.2019-08.basic.20220407T001553Z.RELEASE-2022"
    checkpoint_callback = ModelCheckpoint(
        dirpath='ckpts', 
        filename='harv_10_channels_12_classes_swav_structure_patch_size_3_no_mlp{epoch}',
        every_n_epochs=1,
        save_on_train_epoch_end=True,
        save_top_k = -1
        )


    #IF THIS DOEESNT WORK GO BACK TO TRYING SIMPLE SMALL LOSS OR REMOVING PCA COMPONENTS
    #dataset = HyperDataset(h5_fold, waves=wavelengths, batch_size=256, num_bands=30, crop_size=27)
    # dataset = PreProcDataset(pca_fold, batch_size=256, rearrange=False)
    # train_loader = DataLoader(dataset, batch_size=1, num_workers = 1)
    # #model = models.HyperSimSiamWaveAugment(num_channels=30)
    # #model = models.BYOLTransformer(num_channels=30)
    # model = models.MixedModel(num_channels=30)
    # #model = inference.load_ckpt(models.BYOLTransformer, 'ckpts/harv_transformer_fixed_augment_60_classes_epoch=13.ckpt', num_channels=30)
    # trainer = pl.Trainer(accelerator="cpu", max_epochs=10, callbacks=[checkpoint_callback])
    # trainer.fit(model, train_loader)
    pl.seed_everything(42)
    # dataset = MaskedDenseVitDataset(pca_fold, 32, eval=False, batch_size=8)
    # train_loader = DataLoader(dataset, batch_size=1, num_workers=6)
    # model = models.TransEmbedConvSimSiam(30, img_size=32, output_classes=20) #.load_from_checkpoint('ckpts\harv_trans_embed_conv_sim_epoch=4.ckpt', num_channels=30, img_size=32, output_classes=20)

    dataset = StructureDataset(pca_fold, chm_fold, az_fold, 40)
    train_loader = DataLoader(dataset, batch_size=1, num_workers=4, pin_memory=True)
    model = models.SWaVModelStruct(3, 30)
    trainer = pl.Trainer(accelerator="gpu", max_epochs=200, callbacks=[checkpoint_callback]) #, accumulate_grad_batches=4
    trainer.fit(model, train_loader)

    