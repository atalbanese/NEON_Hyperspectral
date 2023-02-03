from splitting import SiteData
from torch_data import TreeDataset
import pytorch_lightning as pl
import torch
from torch_model import SimpleTransformer
from torch.utils.data import DataLoader


if __name__ == "__main__":
    with torch.autograd.set_detect_anomaly(True):
        test = SiteData(
            site_dir = r'C:\Users\tonyt\Documents\Research\thesis_final\NIWO',
            random_seed=42,
            train = 0.6,
            test= 0.1,
            valid = 0.3)



        test.make_splits('plot_level')
        tree_data = test.get_data('validation', ['hs', 'origin'], 16, make_key=True)

        test_set = TreeDataset(tree_data, output_mode='flat_padded')
        test_loader = DataLoader(test_set, batch_size=8, num_workers=1)
        print(torch.cuda.is_available())

        test_model = SimpleTransformer(
            lr = 5e4,
            emb_size = 512,
            scheduler=True,
            num_features=372,
            num_heads=4,
            num_layers=2,
            num_classes=4,
            sequence_length=16
        )

        trainer = pl.Trainer(accelerator="cpu", max_epochs=1)
        trainer.fit(test_model, test_loader)

