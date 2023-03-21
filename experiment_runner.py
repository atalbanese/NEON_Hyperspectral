import csv
import os
import numpy as np
from splitting import SiteData
from einops import rearrange
from torch_data import PaddedTreeDataSet, SyntheticPaddedTreeDataSet, BasicTreeDataSet
from torch_model import SimpleTransformer
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.ensemble import RandomForestClassifier
import sklearn.metrics as sm

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
        train_prop = 0.6,
        test_prop = 0.2,
        valid_prop = 0.2,
        synth_loader = False,
        num_epochs = 1,
        learning_rate = 5e-4,
        batch_size = 128,
        augments = ['normalize'],
        num_workers = 10,
        ndvi_filter = 0.2,
        mpsi_filter = 0.03,
        sequence_length = 16
    ):
        self.rng = np.random.default_rng(42)
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

        self.synth_loader = synth_loader
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.augments = augments
        self.num_workers = num_workers

        self.site_data = self.gather_data()
        self.training, self.testing, self.validation = self.split_data()      
        self.num_features = self.training[0][self.inp_key].shape[-1]

        self.training_stats = self.calc_stats()
        self.mask_and_pad_data()
        
        #RF doesn't need batched data
        #TBH maybe we could do this for all of them, max batch size
        if self.model_type == 'RF':
            self.batch_size = len(self.training)

        self.train_loader, self.test_loader, self.valid_loader = self.init_dataloaders()

        self.model = self.init_model()
    
    def gather_data(self):
        site_data = SiteData(
            site_dir=os.path.join(self.datadir, self.sitename, self.anno_method, self.man_or_auto),
            random_seed=42,
            train = self.train_prop,
            test = self.test_prop,
            valid = self.valid_prop
        )
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
        collated = self.collate_data(self.training)
        pca = collated['pca']
        mean = np.nanmean(pca, axis=(0,1,2))
        std = np.nanstd(pca, axis=(0,1,2))

        return {'mean': mean, 'std': std}
    
    def init_dataloaders(self):

        if self.split_method == 'pixel':
            return self.init_pixel_loaders()
        
        if self.synth_loader:
            return self.init_synth_loaders()

        return self.init_basic_loaders()

    def init_synth_loaders(self):
        train_set = SyntheticPaddedTreeDataSet(self.training,
                                      pad_length=16,
                                      num_synth_trees=5120,
                                      stats = os.path.join(self.datadir,self.sitename,self.split_method,'stats.npz'),
                                      augments_list=self.augments)
        train_loader = DataLoader(train_set, batch_size=self.batch_size, num_workers=self.num_workers)

        valid_set = PaddedTreeDataSet(self.validation,
                                      pad_length=16,
                                      stats = os.path.join(self.datadir,self.sitename,self.split_method,'stats.npz'),
                                      augments_list=self.augments)
        
        valid_loader = DataLoader(valid_set, batch_size=len(self.validation))

        test_set = PaddedTreeDataSet(self.testing,
                                      pad_length=16,
                                      stats = os.path.join(self.datadir,self.sitename,self.split_method,'stats.npz'),
                                      augments_list=self.augments)
        
        test_loader = DataLoader(test_set, batch_size=len(self.testing))

        return train_loader, test_loader, valid_loader

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

    def init_pixel_loaders(self):
        pass

    def init_model(self):
        if self.model_type == 'RF':
            return self.init_rf_model()
        
        if self.model_type == 'DL':
            return self.init_dl_model()
        
        if self.model_type == 'PT-DL':
            return self.init_pt_dl_model()
    
    def init_rf_model(self):
        return RandomForestClassifier()

    def init_dl_model(self):
        model = SimpleTransformer(
            lr = self.learning_rate,
            emb_size = 128,
            scheduler=True,
            num_features=self.num_features,
            num_heads=4,
            num_layers=12,
            num_classes=len(self.site_data.key.values()),
            weight = list(self.site_data.class_weights.values()),
            classes=self.site_data.key,
            dropout=0.2,
            savedir=self.savedir,
            exp_number=self.exp_number
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
        filename=self.exp_number +'_{val_ova:.2f}_{epoch}',
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
        #TODO: make sure this is 'best' not just last model
        return trainer.test(self.model, dataloaders=self.test_loader)
    
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
    pl.seed_everything(42, workers=True)
    os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
    #TODO: switch to argparse
    savedir = '/home/tony/thesis/lidar_hs_unsup_dl_model/experiment_logs'
    logfile = 'exp_logs.csv'
    datadir = '/home/tony/thesis/lidar_hs_unsup_dl_model/final_data'
    with open('experiments_test.csv') as csvfile:
        with open(logfile, 'w') as csvlog:
            #exp_writer = csv.DictWriter(csvlog)
            exp_reader = csv.DictReader(csvfile)
            exp_writer = csv.DictWriter(csvlog, fieldnames=exp_reader.fieldnames + ['test_ova'])
            exp_writer.writeheader()
            for exp in exp_reader:
                new_exp = Experiment(**exp, savedir=savedir, logfile=logfile, datadir=datadir)
                results = new_exp.run()
                if isinstance(results, list):
                    results = results[0]
                exp['test_ova'] = results['test_ova']
                exp_writer.writerow(exp)

                if 'conf_matrix' in results:
                    rows = []
                    for row in results['conf_matrix']:
                        rows = rows + [[f'{num}' for num in row]]
                    classes = list(new_exp.site_data.key.keys())
                    num_classes = len(classes)
                    with open(os.path.join(savedir,exp['exp_number'],'conf_matrix.csv'), 'w') as conf_file:
                        conf_writer = csv.writer(conf_file)
                        header = ['' for x in range(num_classes+1)]
                        header[1] = 'Expected'
                        conf_writer.writerow(header)
                        header_2 = ['Predicted'] + classes
                        conf_writer.writerow(header_2)
                        for ix, row in enumerate(rows):
                            to_write = [classes[ix]] + row
                            conf_writer.writerow(to_write)



                        

