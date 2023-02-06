from ray import air, tune
from ray.air import session
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
from ray.tune.integration.pytorch_lightning import TuneReportCallback, TuneReportCheckpointCallback
from splitting import SiteData
from torch_data import PaddedTreeDataSet, SyntheticPaddedTreeDataSet
import pytorch_lightning as pl
from torch_model import SimpleTransformer
from torch.utils.data import DataLoader
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint


def train(config, num_epochs):
    pl.seed_everything(42)

    augment_options = {
        0: [],
        1: ["brightness"],
        2: ["brightness", "blit"],
        3: ["brightness", "block"],
        4: ["brightness", "blit", "block"],
        5: ["blit"],
        6: ["blit", "block"],
        7: ["block"]
    }

    niwo = SiteData(
        site_dir = r'C:\Users\tonyt\Documents\Research\thesis_final\NIWO',
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
        stats='stats/niwo_stats.npz',
        augments_list=augment_options[config['augments']] + ["normalize"]
    )
    train_loader = DataLoader(train_set, batch_size=config['batch_size'], num_workers=2)

    valid_set = PaddedTreeDataSet(valid_data, pad_length=16, stats='stats/niwo_stats.npz', augments_list=["normalize"])
    valid_loader = DataLoader(valid_set, batch_size=38)

    train_model = SimpleTransformer(
        lr = config["lr"],
        emb_size = config["emb_size"],
        scheduler=True,
        num_features=372,
        num_heads=12,
        num_layers=6,
        num_classes=4,
        sequence_length=16,
        weight = [1.05,0.744,2.75,0.753],
        classes=niwo.key
    )

    tune_callback = TuneReportCallback(
        {
            "loss": "val_loss",
            "ova": "val_ova"
        },
        on="validation_end",

    )
    

    logger = pl_loggers.TensorBoardLogger(save_dir=r'C:\Users\tonyt\Documents\Research\dl_model\lidar_hs_unsup_dl_model\tuning_data', name="", version=".")
    trainer = pl.Trainer(accelerator="gpu", max_epochs=num_epochs, logger=logger, log_every_n_steps=10, callbacks=[tune_callback])
    trainer.fit(train_model, train_loader, val_dataloaders=valid_loader)

def train_tune(num_samples, num_epochs):
    
    config = {
        "lr": tune.loguniform(1e-5, 1e-1),
        "batch_size": tune.choice([128, 256, 512, 1024, 2048]),
        "emb_size": tune.choice([32, 64, 128, 256, 512, 1024, 2048]),
        "augments": tune.choice([0, 1, 2, 3, 4, 5, 6, 7])
    }

    scheduler = ASHAScheduler(
        max_t=num_epochs,
        grace_period=5,
        reduction_factor=2)

    reporter = CLIReporter(
        parameter_columns=["lr", "batch_size", "emb_size", "augments"],
        metric_columns=["loss", "ova", "training_iteration"])

    train_fn_with_parameters = tune.with_parameters(train,
                                                    num_epochs=num_epochs,)
    resources_per_trial = {"cpu": 2, "gpu": 1}

    tuner = tune.Tuner(
        tune.with_resources(
            train_fn_with_parameters,
            resources=resources_per_trial
        ),
        tune_config=tune.TuneConfig(
            metric="ova",
            mode="min",
            scheduler=scheduler,
            num_samples=num_samples,
        ),
        run_config=air.RunConfig(
            local_dir=r"C:\Users\tonyt\Documents\Research\dl_model\lidar_hs_unsup_dl_model\ray_results",
            name="tune_tree_dl_asha",
            progress_reporter=reporter,
        ),
        param_space=config,
    )
    results = tuner.fit()

    print("Best hyperparameters found were: ", results.get_best_result().config)


if __name__ == "__main__":
    train_tune(50, 50)