from dataloaders import HyperDataset, PreProcDataset, MaskedDataset, MaskedVitDataset, MaskedDenseVitDataset, DeepClusterDataset, StructureDataset
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import models
from pytorch_lightning.callbacks import ModelCheckpoint, StochasticWeightAveraging
import inference
import warnings

def do_training(num_channels=10, num_classes=12, azm=True, chm=True, patch_size=4, log_every=5, max_epochs=50, num_workers=4, img_size=40, extra_labels=''):
    pca_fold = 'C:/Users/tonyt/Documents/Research/datasets/pca/harv_2022_10_channels'
    chm_fold = 'C:/Users/tonyt/Documents/Research/datasets/chm/harv_2019/NEON_struct-ecosystem/NEON.D01.HARV.DP3.30015.001.2019-08.basic.20220511T165943Z.RELEASE-2022'
    az_fold = 'C:/Users/tonyt/Documents/Research/datasets/solar_azimuth/harv_2022'

    checkpoint_callback = ModelCheckpoint(
        dirpath='ckpts', 
        filename=f'harv_{num_channels}_channels_{num_classes}_classes_swav_structure_patch_size_{patch_size}_{extra_labels}'+'_{epoch}',
        every_n_epochs=log_every,
        save_on_train_epoch_end=True,
        save_top_k = -1
        )

    pl.seed_everything(42)

    dataset = StructureDataset(pca_fold, chm_fold, az_fold, img_size)
    train_loader = DataLoader(dataset, batch_size=1, num_workers=num_workers, pin_memory=True)
    model = models.SWaVModelStruct(patch_size, img_size, azm=azm, chm=chm)
    trainer = pl.Trainer(accelerator="gpu", max_epochs=max_epochs, callbacks=[checkpoint_callback]) #, accumulate_grad_batches=4
    trainer.fit(model, train_loader)


if __name__ == "__main__":

    configs = [
        {'num_channels': 10,
         'num_classes': 12,
         'azm': True,
         'chm': True,
         'patch_size': 4,
         'log_every': 10,
         'max_epochs': 50,
         'num_workers': 4,
         'img_size': 40,
         'extra_labels': 'chm_and_azm'
         },
          {'num_channels': 10,
         'num_classes': 12,
         'azm': False,
         'chm': True,
         'patch_size': 4,
         'log_every': 10,
         'max_epochs': 50,
         'num_workers': 4,
         'img_size': 40,
         'extra_labels': 'azm_only'
         },
          {'num_channels': 10,
         'num_classes': 12,
         'azm': False,
         'chm': True,
         'patch_size': 4,
         'log_every': 10,
         'max_epochs': 50,
         'num_workers': 4,
         'img_size': 40,
         'extra_labels': 'chm_only'
         }
    ]

    for config in configs:
        do_training(**config)

    