from torch_pretraining_model import PreTrainingModel
from pretraining_data import PreTrainingData
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning import loggers as pl_loggers

if __name__ == "__main__":
    pl.seed_everything(42)

    pre_data = PreTrainingData(
        data_dir="/home/tony/thesis/data/NIWO_unlabeled",
        sequence_length=16,
        hs_filters=[[410,1357],[1400,1800],[1965,2485]],
        sitename="NIWO",
        augments_list=["normalize"],
        file_dim=1000,
        stats="/home/tony/thesis/data/stats/niwo_stats.npz"
    )

    train_loader = DataLoader(pre_data, batch_size=2048, num_workers=10)

    train_model = PreTrainingModel(
        lr=5e-4,
        emb_size=128,
        scheduler=True,
        num_features=372,
        sequence_length=16,
        num_heads=12,
        num_layers=4
    )

    exp_name = "Pre_Training_Test_1"
    logger = pl_loggers.TensorBoardLogger(save_dir = './pre_training_logs', name=exp_name)
    trainer = pl.Trainer(accelerator="gpu", max_epochs=15, logger=logger,)

    trainer.fit(train_model, train_loader)
    trainer.save_checkpoint('/home/tony/thesis/pre_training_ckpts/pre_training_1.ckpt')

