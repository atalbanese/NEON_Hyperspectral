from splitting import SiteData
from torch_data import PaddedTreeDataSet, SyntheticPaddedTreeDataSet
import pytorch_lightning as pl
#import torch
from torch_model import SimpleTransformer
from torch.utils.data import DataLoader
from pytorch_lightning import loggers as pl_loggers



if __name__ == "__main__":

    niwo = SiteData(
        site_dir = r'C:\Users\tonyt\Documents\Research\thesis_final\NIWO',
        random_seed=42,
        train = 0.6,
        test= 0.3,
        valid = 0.1)



    niwo.make_splits('plot_level')
    tree_data = niwo.get_data('training', ['hs', 'origin'], 16, make_key=True)
    valid_data = niwo.get_data('validation', ['hs', 'origin'], 16, make_key=True)

    

    #train_set = PaddedTreeDataSet(tree_data, pad_length=16)
    train_set = SyntheticPaddedTreeDataSet(
        tree_list=tree_data,
        pad_length=16,
        num_synth_trees=2560,
        num_features=372
    )


    test_loader = DataLoader(train_set, batch_size=256, num_workers=2)

    valid_set = PaddedTreeDataSet(valid_data, pad_length=16)
    valid_loader = DataLoader(valid_set, batch_size=len(valid_set))

    test_model = SimpleTransformer(
        lr = 5e-4,
        emb_size = 512,
        scheduler=True,
        num_features=372,
        num_heads=12,
        num_layers=6,
        num_classes=4,
        sequence_length=16
    )

    tb_logger = pl_loggers.TensorBoardLogger(save_dir="thesis_final_logs/")
    trainer = pl.Trainer(accelerator="gpu", max_epochs=5000, logger=tb_logger, log_every_n_steps=10)
    trainer.fit(test_model, test_loader, val_dataloaders=valid_loader)

