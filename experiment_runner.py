import csv
import os
import numpy as np
from splitting import SiteData
from einops import rearrange
from torch_data import PixelTreeDataSet, BasicTreeDataSet
from torch_model import SimpleTransformer
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.ensemble import RandomForestClassifier
import sklearn.metrics as sm
import traceback
import argparse

class Experiment:
    def __init__(self,
        exp_number,
        sitename,
        anno_method,
        man_or_auto,
        split_method,
        model,
        savedir,
        logfile,
        datadir,
        apply_filters,
        inp_key,
        trial_num,
        train_prop = 0.6,
        test_prop = 0.2,
        valid_prop = 0.2,
        synth_loader = False,
        num_epochs = 200,
        learning_rate = 5e-4,
        batch_size = 128,
        augments = ['normalize'],
        num_workers = 10,
        ndvi_filter = 0.2,
        mpsi_filter = 0.03,
        sequence_length = 16,
        data_dim = 4,
        remove_taxa = '',
        **kwargs,
    ):
        self.rng = np.random.default_rng()
        self.data_dim = data_dim
        self.trial_num = trial_num
        self.exp_number = exp_number
        self.sitename = sitename
        self.anno_method = anno_method
        self.man_or_auto = man_or_auto
        self.split_method = split_method
        self.model_type = model
        self.savedir = savedir
        self.logfile = logfile
        self.datadir = datadir
        self.train_prop = train_prop
        self.test_prop = test_prop
        self.valid_prop = valid_prop
        self.ndvi_filter = ndvi_filter
        self.mpsi_filter = mpsi_filter
        self.apply_filters = True if apply_filters == 'T' else False
        self.inp_key = inp_key
        self.sequence_length = sequence_length
        if len(remove_taxa)>0:
            self.remove_taxa = remove_taxa.split(';')
        else:
            self.remove_taxa = []

        self.synth_loader = synth_loader
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.augments = augments
        self.num_workers = num_workers

        self.site_data = self.gather_data()
        self.training, self.testing, self.validation = self.split_data()
        if split_method != 'pixel':      
            self.num_features = self.training[0][self.inp_key].shape[-1]
        else:
            self.num_features = self.training[self.inp_key].shape[-1]

        self.training_stats = self.calc_stats()
        if split_method != 'pixel':
            self.mask_and_pad_data()
        else:
            self.mask_pixel()
        
        #RF doesn't need batched data
        #TBH maybe we could do this for all of them, max batch size
        if self.model_type == 'RF':
            self.batch_size = len(self.training)

        self.train_loader, self.test_loader, self.valid_loader = self.init_dataloaders()

        self.model = self.init_model()
    
    def gather_data(self):
        site_data = SiteData(
            site_dir=os.path.join(self.datadir, self.sitename, self.anno_method, self.man_or_auto),
            train = self.train_prop,
            test = self.test_prop,
            valid = self.valid_prop,
            out_dim = self.data_dim,
            apply_filters=self.apply_filters,
            taxa_to_drop=self.remove_taxa
        )
        site_data.taxa_plot_counts
        if self.split_method != 'pixel':
            site_data.make_splits(self.split_method)

        return site_data
    
    def split_data(self):
        if self.split_method != 'pixel':
            train = self.site_data.get_data('training', ['hs', 'pca', 'chm', 'ndvi', 'mpsi'], make_key=True)
            test = self.site_data.get_data('testing', ['hs', 'pca', 'chm', 'ndvi', 'mpsi'],make_key=True)
            valid = self.site_data.get_data('validation', ['hs', 'pca', 'chm', 'ndvi', 'mpsi'],  make_key=True)

            return train, test, valid
        else:
            return self.get_pixel_split()
    
    #This should really be part of data splitter but somnehow this is easier.
    def get_pixel_split(self):
        all_data = self.site_data.get_data('all', ['hs', 'pca', 'chm', 'ndvi', 'mpsi'], make_key=True)
        collated = self.collate_data(all_data)
        flattened = dict()
        length = 0
        for k, v in collated.items():
            if len(v.shape) == 4:
                flat = rearrange(v, 'b h w c -> (b h w) c')
                flattened[k] = flat
            if len(v.shape) == 3:
                flat = rearrange(v, 'b h w -> (b h w)')
                flattened[k] = flat
                length = flat.shape[0]

        #Splitting samples
        num_train = int(length*self.train_prop)
        num_valid = int(length*self.valid_prop)
        all_idxs = np.arange(length)

        
        train_samples = self.rng.choice(all_idxs, size=num_train, replace=False)
        remain_idxs = np.setdiff1d(all_idxs, train_samples)
        valid_samples = self.rng.choice(remain_idxs, size=num_valid, replace=False)
        test_samples = np.setdiff1d(remain_idxs, valid_samples)
        train, test, valid = dict(), dict(), dict()

        for k,v in flattened.items():
            train[k] = v[train_samples]
            test[k] = v[test_samples]
            valid[k] = v[valid_samples]
        
        #Swapping valid and test to avoid getting a test set where there aren't pixels from one class. Happens with NIWO/Scholl
        #TODO: Fix sampling to ensure this can't happen
        return train, test, valid

    def collate_data(self, data_list):
        keys = data_list[0].keys()
        data_holder = {k:[] for k in keys}
        for entry in data_list:
            for k, v in entry.items():
                data_holder[k].append(v)
        out_dict = dict()
        for k,v in data_holder.items():
            out_dict[k] = np.stack(v)

        return out_dict

    def calc_stats(self):
        if self.split_method != 'pixel':
            collated = self.collate_data(self.training)
            inp = collated[self.inp_key]
            mean = np.nanmean(inp, axis=(0,1,2))
            std = np.nanstd(inp, axis=(0,1,2))
        else:
            mean = np.nanmean(self.training[self.inp_key], axis=0)
            std = np.nanstd(self.training[self.inp_key], axis=0)
        #Hack to prevent divide by 0 in case there is a channel with 0 standard deviation - it occured once I swear it!
        std[std == 0] = 0.00001
        return {'mean': mean, 'std': std}
    
    def init_dataloaders(self):

        if self.split_method == 'pixel':
            return self.init_pixel_loaders()
        
        if self.synth_loader:
            return self.init_synth_loaders()

        return self.init_basic_loaders()

    def init_synth_loaders(self):
        pass

    def init_basic_loaders(self):
        train_set = BasicTreeDataSet(self.training,
                                      stats = self.training_stats,
                                      augments_list=self.augments,
                                      inp_key=self.inp_key)
        train_loader = DataLoader(train_set, batch_size=self.batch_size, num_workers=self.num_workers)

        valid_set = BasicTreeDataSet(self.validation,
                                      stats = self.training_stats,
                                      augments_list=self.augments,
                                      inp_key=self.inp_key)
        
        valid_loader = DataLoader(valid_set, batch_size=len(self.validation))

        test_set = BasicTreeDataSet(self.testing,
                                      stats = self.training_stats,
                                      augments_list=self.augments,
                                      inp_key=self.inp_key)
        
        test_loader = DataLoader(test_set, batch_size=len(self.testing))

        return train_loader, test_loader, valid_loader

    def mask_and_pad_data(self):
        for dset in [self.training, self.testing, self.validation]:
            for ix, tree in enumerate(dset):
                chm_mask = tree['chm'] > 1.99

                if self.apply_filters:
                    ndvi_mask = tree['ndvi'] > self.ndvi_filter
                    mpsi_mask = tree['mpsi'] > self.mpsi_filter
                    chm_mask = chm_mask * ndvi_mask * mpsi_mask
                pad_dif = np.count_nonzero(~chm_mask)
                tree[self.inp_key] = np.pad(tree[self.inp_key][chm_mask], ((0, pad_dif), (0,0)))
                tree['pixel_target'] = np.pad(tree['pixel_target'][chm_mask], ((0, pad_dif)))
                pad_mask = np.zeros((self.sequence_length,), dtype=np.bool_)
                if pad_dif > 0:
                    pad_mask[-pad_dif:] = True
                tree['pad_mask'] = pad_mask
                tree['mask'] = chm_mask

    def mask_pixel(self):
        for dset in [self.training, self.testing, self.validation]:
            chm_mask = dset['chm'] > 1.99
            if self.apply_filters:
                ndvi_mask = dset['ndvi'] > self.ndvi_filter
                mpsi_mask = dset['mpsi'] > self.mpsi_filter
                chm_mask = chm_mask * ndvi_mask * mpsi_mask
            dset[self.inp_key] = dset[self.inp_key][chm_mask]
            dset['pixel_target'] = dset['pixel_target'][chm_mask]

    def init_pixel_loaders(self):
        train_set = PixelTreeDataSet(self.training,
                                      stats = self.training_stats,
                                      augments_list=self.augments,
                                      inp_key=self.inp_key,
                                      sequence_length=self.sequence_length)

        batch_size = train_set.__len__() if self.model_type == 'RF' else self.batch_size
        train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=self.num_workers)

        valid_set = PixelTreeDataSet(self.validation,
                                      stats = self.training_stats,
                                      augments_list=self.augments,
                                      inp_key=self.inp_key,
                                      sequence_length=self.sequence_length)
        
        valid_loader = DataLoader(valid_set, batch_size=valid_set.__len__())

        test_set = PixelTreeDataSet(self.testing,
                                      stats = self.training_stats,
                                      augments_list=self.augments,
                                      inp_key=self.inp_key,
                                      sequence_length=self.sequence_length)
        
        test_loader = DataLoader(test_set, batch_size=test_set.__len__())
        return train_loader, test_loader, valid_loader

    def init_model(self):
        if self.model_type == 'RF':
            return self.init_rf_model()
        
        if self.model_type == 'DL':
            return self.init_dl_model()
        
        if self.model_type == 'PT-DL':
            return self.init_pt_dl_model()
    
    def init_rf_model(self):
        return RandomForestClassifier()

    def pixel_weights(self):
        class_weights = dict()
        n_classes = len(self.site_data.all_taxa.keys())
        n_samples = self.training[self.inp_key].shape[0]
        for k, v in self.site_data.key.items():
            class_count = (self.training['pixel_target'] == v).sum()
            class_weights[k] = n_samples/(n_classes*class_count)
        return list(class_weights.values())


    def init_dl_model(self):
        if self.split_method != 'pixel':
            weight = list(self.site_data.class_weights.values())
        else:
            weight = self.pixel_weights()

        num_heads = 4 if self.inp_key == 'pca' else 12
        model = SimpleTransformer(
            lr = self.learning_rate,
            emb_size = 128,
            scheduler=True,
            num_features=self.num_features,
            num_heads=num_heads,
            num_layers=12,
            num_classes=len(self.site_data.key.values()),
            weight = weight,
            classes=self.site_data.key,
            dropout=0.2,
            savedir=self.savedir,
            exp_number=self.exp_number,
            trial_number=self.trial_num
            )
        return model

    def init_pt_dl_model(self):
        pass

    def run(self):
        if self.model_type == 'RF':
            return self.run_rf_model()
        
        if self.model_type == 'DL':
            return self.run_dl_model()
        
        if self.model_type == 'PT-DL':
            return self.run_pt_dl_model()

    def run_dl_model(self):

        val_callback = ModelCheckpoint(
        dirpath=os.path.join(self.datadir,'ckpts'), 
        filename=f'{self.sitename}_exp_{self.exp_number}_trial_{self.trial_num}' +'_{val_ova:.2f}_{epoch}',
        monitor='val_loss',
        save_on_train_epoch_end=True,
        mode='min',
        save_top_k = 3
        )


        logger = pl_loggers.TensorBoardLogger(save_dir = self.savedir, name=self.exp_number)
        trainer = pl.Trainer(accelerator="gpu", max_epochs=self.num_epochs, logger=logger, log_every_n_steps=10, deterministic=True,
            callbacks=[
            val_callback
            ]
            )
        trainer.fit(self.model, self.train_loader, val_dataloaders=self.valid_loader)
        return trainer.test(self.model, dataloaders=self.test_loader, ckpt_path='best')
    
    def run_rf_model(self):
        #Somewhat ridiculous but using pytorch dataloaders to collate + normalize the data because the code is in place to do so
        #RF and DL would be better as separate Experiments but I'm short on time here
        for train in self.train_loader:
            pass
        for valid in self.valid_loader:
            pass
        for test in self.test_loader:
            pass

        inp = np.concatenate((train['input'][~train['pad_mask']].numpy(), valid['input'][~valid['pad_mask']].numpy()))

        inp_targets = np.concatenate((train['target'][~train['pad_mask']].numpy(), valid['target'][~valid['pad_mask']].numpy()))

        test_inp = test['input'][~test['pad_mask']].numpy()

        test_targets = test['target'][~test['pad_mask']].numpy()

        self.model.fit(inp, inp_targets)
        predictions = self.model.predict(test_inp)
        conf_matrix = sm.confusion_matrix(test_targets, predictions)
        print(conf_matrix)
        ova = self.model.score(test_inp, test_targets)
        return {'test_ova': ova, 'conf_matrix': conf_matrix}
        


if __name__ == '__main__':

    os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'


    parser = argparse.ArgumentParser()
    parser.add_argument("savedir", help='Directory to save DL logs + Confusion matrices (subdirs will be generated per experiment number)', type=str)
    parser.add_argument("logfile", help="File to log experiment results", type=str)
    parser.add_argument("datadir", help='Base directory storing all NEON data', type=str)
    parser.add_argument('exp_file',  help='CSV file containing experiments to run', type=str)
    parser.add_argument("-f", "--fixed_seed", help="Use a fixed seed for all rngs", action="store_true")

    args = parser.parse_args()

    if args.fixed_seed:
        pl.seed_everything(42, workers=True)

    datadir = args.datadir
    logfile = args.logfile
    savedir = args.savedir
    exp_file = args.exp_file

    #pl.seed_everything(42, workers=True)
    # savedir = '/home/tony/thesis/lidar_hs_unsup_dl_model/experiment_logs'
    # logfile = 'exp_logs.csv'
    # datadir = '/home/tony/thesis/lidar_hs_unsup_dl_model/final_data'
    # exp_file = '/home/tony/thesis/lidar_hs_unsup_dl_model/experiments_test.csv'

    with open(exp_file) as csvfile:
        with open(logfile, 'w') as csvlog:
            exp_reader = csv.DictReader(csvfile)
            fixed_names = [fname.replace('num_trials', 'trial_num') for fname in exp_reader.fieldnames] + ['test_ova']
            exp_writer = csv.DictWriter(csvlog, fieldnames=fixed_names)
            exp_writer.writeheader()
            csvlog.flush()
            for exp in exp_reader:
                exp['num_trials'] = int(exp['num_trials'])
                for trial in range(exp['num_trials']):
                    
                    try:
                        exp['trial_num'] = trial + 1
                        del exp['test_ova']
                        print(f'Starting trial {exp["trial_num"]} of experiment {exp["exp_number"]} with params: {exp}')
                        new_exp = Experiment(**exp, savedir=savedir, logfile=logfile, datadir=datadir)
                        results = new_exp.run()
                        if isinstance(results, list):
                            results = results[0]
                        exp['test_ova'] = results['test_ova']
                        exp_line = exp.copy()
                        del exp_line['num_trials']
                        exp_writer.writerow(exp_line)
                        csvlog.flush()

                        if 'conf_matrix' in results:
                            rows = []
                            for row in results['conf_matrix']:
                                rows = rows + [[f'{num}' for num in row]]
                            classes = list(new_exp.site_data.key.keys())
                            num_classes = len(classes)
                            with open(os.path.join(savedir,f'{exp["exp_number"]}_{exp["trial_num"]}_conf_matrix.csv'), 'w') as conf_file:
                                conf_writer = csv.writer(conf_file)
                                header = ['' for x in range(num_classes+1)]
                                header[1] = 'Expected'
                                conf_writer.writerow(header)
                                header_2 = ['Predicted'] + classes
                                conf_writer.writerow(header_2)
                                for ix, row in enumerate(rows):
                                    to_write = [classes[ix]] + row
                                    conf_writer.writerow(to_write)
                    except Exception as e:
                        print(e)
                        traceback.print_exc()
                        with open(os.path.join(savedir, 'error_logs.txt'), 'a') as f:
                            f.write(f'Error running exp: {exp["exp_number"]} trial: {exp["trial_num"]}\n')
                            f.write(str(e))
                            f.write('\n')




                        

