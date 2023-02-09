from splitting import SiteData
from torch_data import PaddedTreeDataSet, SyntheticPaddedTreeDataSet
import pytorch_lightning as pl
#import torch
from torch_model import SimpleTransformer, SimpleLinearModel
from torch.utils.data import DataLoader
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint



if __name__ == "__main__":
    pl.seed_everything(42, workers=True)
    
    niwo = SiteData(
        site_dir = '/home/tony/thesis/data/NIWO',
        random_seed=42,
        train = 0.6,
        test= 0.3,
        valid = 0.1)



    niwo.make_splits('plot_level')
    train_data = niwo.get_data('training', ['hs', 'origin'], 16, make_key=True)
    valid_data = niwo.get_data('validation', ['hs', 'origin'], 16, make_key=True)
    test_data = niwo.get_data('testing', ['hs', 'origin'], 16, make_key=True)

    


    train_set = SyntheticPaddedTreeDataSet(
        tree_list=train_data,
        pad_length=16,
        num_synth_trees=5120,
        num_features=372,
        stats='/home/tony/thesis/data/stats/niwo_stats.npz',
        augments_list=[ "normalize"],
        weights=list(niwo.class_weights.values())
        #weights=None
    )
    #train_set = PaddedTreeDataSet(train_data, pad_length=16, stats='/home/tony/thesis/data/stats/niwo_stats.npz', augments_list=["brightness", "blit", "block", "normalize"])
    train_loader = DataLoader(train_set, batch_size=512, num_workers=10)

    valid_set = PaddedTreeDataSet(valid_data, pad_length=16, stats='/home/tony/thesis/data/stats/niwo_stats.npz', augments_list=["normalize"])
    valid_loader = DataLoader(valid_set, batch_size=38)

    test_set = PaddedTreeDataSet(test_data, pad_length=16, stats='/home/tony/thesis/data/stats/niwo_stats.npz', augments_list=["normalize"])
    test_loader = DataLoader(test_set, batch_size = len(test_set))

    train_model = SimpleTransformer(
        lr = 5e-5,
        emb_size = 1024,
        scheduler=True,
        num_features=372,
        num_heads=4,
        num_layers=6,
        num_classes=4,
        sequence_length=16,
        weight = list(niwo.class_weights.values()),
        #weight=None,
        classes=niwo.key
    )

    # train_model=SimpleLinearModel(
    #     lr =5e-4,
    #     scheduler=True,
    #     num_features=372,
    #     num_classes=4,
    #     sequence_length=16,
    #     weight=[1.05,0.744,2.75,0.753],
    #     #weight=None,
    #     classes=niwo.key
    # )

    exp_name = 'samp_weights_val_loss_min'
    val_callback = ModelCheckpoint(
        dirpath='ckpts/', 
        filename=exp_name +'{val_ova:.2f}_{epoch}',
        monitor='val_loss',
        save_on_train_epoch_end=True,
        mode='min',
        save_top_k = 3
        )

    logger = pl_loggers.TensorBoardLogger(save_dir = './tuning_logs', name=exp_name)
    trainer = pl.Trainer(accelerator="gpu", max_epochs=20, logger=logger, log_every_n_steps=10, deterministic=True,
    #callbacks=[val_callback]
    )
    #trainer.tune(train_model, train_loader, val_dataloaders=valid_loader)
    trainer.fit(train_model, train_loader, val_dataloaders=valid_loader)
    trainer.test(train_model, dataloaders=test_loader)

