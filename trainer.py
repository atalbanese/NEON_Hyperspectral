from dataloaders import HyperDataset
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import models

if __name__ == "__main__":
    h5_fold = "/data/shared/src/aalbanese/datasets/hs/NEON_refl-surf-dir-ortho-mosaic/NEON.D16.WREF.DP3.30006.001.2021-07.basic.20220330T192306Z.PROVISIONAL"

    #TODO: Make these single wavelengths
    wavelengths = {"red": 654,
                    "green": 561, 
                    "blue": 482,
                    "nir": 865}

    dataset = HyperDataset(h5_fold, waves=wavelengths, augment="wavelength", batch_size=64)
    train_loader = DataLoader(dataset, batch_size=1, num_workers = 30)
    model = models.HyperSimSiamWaveAugment(num_channels=4)
    trainer = pl.Trainer(accelerator="cpu", max_epochs=20)
    trainer.fit(model, train_loader)

    