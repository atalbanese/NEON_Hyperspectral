from splitting import SiteData
from torch_data import TreeDataset
import pytorch_lightning as pl
import torch
from torch_model import SimpleTransformer
from torch.utils.data import DataLoader


if __name__ == "__main__":
    with torch.autograd.set_detect_anomaly(True):
        niwo = SiteData(
            site_dir = r'C:\Users\tonyt\Documents\Research\thesis_final\NIWO',
            random_seed=42,
            train = 0.6,
            test= 0.3,
            valid = 0.1)



        niwo.make_splits('plot_level')
        tree_data = niwo.get_data('training', ['hs', 'origin'], 16, make_key=True)
        valid_data = niwo.get_data('validation', ['hs', 'origin'], 16, make_key=True)

        

        train_set = TreeDataset(tree_data, output_mode='flat_padded')
        test_loader = DataLoader(train_set, batch_size=64, num_workers=4)

        valid_set = TreeDataset(valid_data, output_mode='flat_padded')
        valid_loader = DataLoader(valid_set, batch_size=len(valid_set))

        test_model = SimpleTransformer(
            lr = 5e-4,
            emb_size = 512,
            scheduler=True,
            num_features=372,
            num_heads=6,
            num_layers=4,
            num_classes=4,
            sequence_length=16
        )

        trainer = pl.Trainer(accelerator="gpu", max_epochs=500)
        trainer.fit(test_model, test_loader, val_dataloaders=valid_loader)

