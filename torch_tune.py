import optuna
import os
from optuna.integration import PyTorchLightningPruningCallback
from splitting import SiteData
from torch_data import PaddedTreeDataSet, SyntheticPaddedTreeDataSet
import pytorch_lightning as pl
from torch_model import SimpleTransformer
from torch.utils.data import DataLoader
from pytorch_lightning import loggers as pl_loggers


EPOCHS = 100


def objective(trial: optuna.trial.Trial):
    lr = trial.suggest_float('lr', 1e-5, 1e-4, log=True)
    batch_size = trial.suggest_int('batch_size', 128, 1024)
    emb_size = trial.suggest_int("emb_size", 350, 512)

    niwo = SiteData(
        site_dir = '/home/tony/thesis/data/NIWO',
        random_seed=42,
        train = 0.6,
        test= 0.3,
        valid = 0.1)

    niwo.make_splits('plot_level')
    train_data = niwo.get_data('training', ['hs', 'origin'], 16, make_key=True)
    valid_data = niwo.get_data('validation', ['hs', 'origin'], 16, make_key=True)
    
    
    train_set = SyntheticPaddedTreeDataSet(
        tree_list=train_data,
        pad_length=16,
        num_synth_trees=5120,
        num_features=372,
        stats='/home/tony/thesis/data/stats/niwo_stats.npz',
        augments_list=["normalize"]
    )
    train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=4)

    valid_set = PaddedTreeDataSet(valid_data, pad_length=16, stats='/home/tony/thesis/data/stats/niwo_stats.npz', augments_list=["normalize"])
    valid_loader = DataLoader(valid_set, batch_size=38)

    train_model = SimpleTransformer(
        lr = lr,
        emb_size = emb_size,
        scheduler=True,
        num_features=372,
        num_heads=12,
        num_layers=6,
        num_classes=4,
        sequence_length=16,
        weight = [1.05,0.744,2.75,0.753],
        classes=niwo.key
    )

    trainer = pl.Trainer(accelerator='gpu', max_epochs=EPOCHS, log_every_n_steps=10, callbacks=[PyTorchLightningPruningCallback(trial, monitor="val_loss")])
    hyperparameters = dict(lr=lr, batch_size=batch_size, emb_size=emb_size)
    trainer.logger.log_hyperparams(hyperparameters)
    trainer.fit(train_model, train_dataloaders=train_loader, val_dataloaders=valid_loader)

    return trainer.callback_metrics["val_loss"].item()







if __name__ == "__main__":
    pruner = optuna.pruners.MedianPruner()
    study = optuna.create_study(direction="minimize", pruner=pruner)
    study.optimize(objective, n_trials=100)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    df = study.trials_dataframe()

    df.to_csv(os.path.join(os.getcwd(), "trials.csv"))