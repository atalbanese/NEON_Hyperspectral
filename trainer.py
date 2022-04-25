from dataloaders import HyperDataset, PreProcDataset
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import models
from pytorch_lightning.callbacks import ModelCheckpoint
import inference


if __name__ == "__main__":
    pca_fold = '/data/shared/src/aalbanese/datasets/hs/pca/harv_2022'
    #h5_fold = "/data/shared/src/aalbanese/datasets/hs/NEON_refl-surf-dir-ortho-mosaic/NEON.D16.WREF.DP3.30006.001.2021-07.basic.20220330T192306Z.PROVISIONAL"
    h5_fold = "/data/shared/src/aalbanese/datasets/hs/NEON_refl-surf-dir-ortho-mosaic/NEON.D01.HARV.DP3.30006.001.2019-08.basic.20220407T001553Z.RELEASE-2022"
    checkpoint_callback = ModelCheckpoint(
        dirpath='ckpts', 
        filename='harv_transformer_fixed_augment_60_classes_{epoch}',
        every_n_epochs=1,
        save_on_train_epoch_end=True,
        save_top_k = -1
        )

    #TODO: Make these single wavelengths
    # wavelengths = {"red": 654,
    #                 "green": 561, 
    #                 "blue": 482,
    #                 "nir": 865}

    wavelengths = {f"{i}": i for i in range(50, 1200, 100)}

    #dataset = HyperDataset(h5_fold, waves=wavelengths, batch_size=256, num_bands=30, crop_size=27)
    dataset = PreProcDataset(pca_fold, batch_size=256, rearrange=False)
    train_loader = DataLoader(dataset, batch_size=1, num_workers = 1)
    #model = models.HyperSimSiamWaveAugment(num_channels=30)
    #model = models.BYOLTransformer(num_channels=30)
    model = models.MixedModel(num_channels=30)
    #model = inference.load_ckpt(models.BYOLTransformer, 'ckpts/harv_transformer_fixed_augment_60_classes_epoch=13.ckpt', num_channels=30)
    trainer = pl.Trainer(accelerator="cpu", max_epochs=25, callbacks=[checkpoint_callback])
    trainer.fit(model, train_loader)

    