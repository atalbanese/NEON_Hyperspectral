from dataloaders import HyperDataset
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from models import HyperSimSiam

if __name__ == "__main__":
    h5_fold = "/data/shared/src/aalbanese/datasets/hs/NEON_refl-surf-dir-ortho-mosaic/NEON.D16.WREF.DP3.30006.001.2021-07.basic.20220330T192306Z.PROVISIONAL"

    #TODO: Make these single wavelengths
    wavelengths = {"blue": {"lower": 452, "upper": 512},
                     "green": {"lower": 533, "upper": 590},
                     "red": {"lower": 636, "upper": 673}}

    dataset = HyperDataset(h5_fold, waves=wavelengths)
    train_loader = DataLoader(dataset, batch_size=1, num_workers = 20)
    model = HyperSimSiam(num_channels=3)
    trainer = pl.Trainer(accelerator="cpu")
    trainer.fit(model, train_loader)

    