from torch_pretraining_model import PreTrainingModel
from pretraining_data import PreTrainingData
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning import loggers as pl_loggers
import argparse
import os

if __name__ == "__main__":
    #pl.seed_everything(42)
    os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
    parser = argparse.ArgumentParser()
    parser.add_argument("savedir", help='Directory to save pre-training ckpts', type=str)
    parser.add_argument("datadir", help='Base directory storing all NEON data', type=str)
    parser.add_argument('sitename',  help='Name of NEON site to pre-train on', type=str)

    parser.add_argument("-f", "--fixed_seed", help="Use a fixed seed for all rngs", action="store_true")

    args = parser.parse_args()

    if args.fixed_seed:
        pl.seed_everything(42, workers=True)

    datadir = args.datadir
    sitename = args.sitename
    savedir=args.savedir

    # datadir = '/home/tony/thesis/lidar_hs_unsup_dl_model/final_data/'
    # sitename = 'STEI'
    # savedir = '/home/tony/thesis/lidar_hs_unsup_dl_model/final_data/pre_training_ckpts'

    pre_data = PreTrainingData(
        data_dir=os.path.join(datadir, sitename, 'PCA'),
        sequence_length=16,
        sitename=sitename,
        augments_list=["normalize"],
        file_dim=1000,
    )

    train_loader = DataLoader(pre_data, batch_size=1024, num_workers=10)

    example_item = pre_data.__getitem__(1)

    train_model = PreTrainingModel(
        lr=5e-5,
        emb_size=128,
        scheduler=True,
        num_features=example_item['input'].shape[-1],
        sequence_length=16,
        num_heads=4,
        num_layers=12
    )

    exp_name = f"Pre_Training_{sitename}"
    logger = pl_loggers.TensorBoardLogger(save_dir = './pre_training_logs', name=exp_name)
    trainer = pl.Trainer(accelerator="gpu", max_epochs=15, logger=logger,)

    trainer.fit(train_model, train_loader)
    trainer.save_checkpoint(os.path.join(savedir, f'pre_training_{sitename}.ckpt'))

