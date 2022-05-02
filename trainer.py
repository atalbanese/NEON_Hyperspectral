from dataloaders import HyperDataset, PreProcDataset, MaskedDataset, MaskedVitDataset, MaskedDenseVitDataset
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import models
from pytorch_lightning.callbacks import ModelCheckpoint, StochasticWeightAveraging
import inference


if __name__ == "__main__":
    pca_fold = 'C:/Users/tonyt/Documents/Research/datasets/pca/harv_2022'
    #h5_fold = "/data/shared/src/aalbanese/datasets/hs/NEON_refl-surf-dir-ortho-mosaic/NEON.D16.WREF.DP3.30006.001.2021-07.basic.20220330T192306Z.PROVISIONAL"
    h5_fold = "/data/shared/src/aalbanese/datasets/hs/NEON_refl-surf-dir-ortho-mosaic/NEON.D01.HARV.DP3.30006.001.2019-08.basic.20220407T001553Z.RELEASE-2022"
    checkpoint_callback = ModelCheckpoint(
        dirpath='ckpts', 
        filename='harv_sim_siam_patched_{epoch}',
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

    dataset = MaskedDenseVitDataset(pca_fold, 64, eval=False, batch_size=8)
    train_loader = DataLoader(dataset, batch_size=1, num_workers=4)
    model = models.PatchedSimSiam(30, img_size=64, patch_size=4) #.load_from_checkpoint('ckpts/harv_sim_siam_masked_patched_vit_dense_vizepoch=2-v1.ckpt', num_channels=30, img_size=64, patch_size=4)

    trainer = pl.Trainer(accelerator="gpu", max_epochs=10, callbacks=[checkpoint_callback, StochasticWeightAveraging(swa_lrs=1e-2)], accumulate_grad_batches=8)
    trainer.fit(model, train_loader)

    