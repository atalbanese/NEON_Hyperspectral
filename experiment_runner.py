import csv
import os
from splitting import SiteData



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
        train_prop = 0.6,
        test_prop = 0.2,
        valid_prop = 0.2
    ):
        self.exp_number = exp_number
        self.sitename = sitename
        self.anno_method = anno_method
        self.man_or_auto = man_or_auto
        self.split_method = split_method
        self.model = model
        self.savedir = savedir
        self.logfile = logfile
        self.datadir = datadir
        self.train_prop = train_prop
        self.test_prop = test_prop
        self.valid_prop = valid_prop

        self.site_data = self.gather_data()

        self.training, self.testing, self.validation = self.split_data()
    
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
            train = self.site_data.get_data('training', ['hs', 'pca', 'chm', 'ndvi', 'mpsi'], 16, make_key=True)
            test = self.site_data.get_data('testing', ['hs', 'pca', 'chm', 'ndvi', 'mpsi'], 16, make_key=True)
            valid = self.site_data.get_data('validation', ['hs', 'pca', 'chm', 'ndvi', 'mpsi'], 16, make_key=True)

            return train, test, valid
        else:
            return self.get_pixel_split()
        pass
    
    #This should really be part of data splitter but somnehow this is easier. Pixel splitting is just plain wrong but science demands I test it
    def get_pixel_split(self):
        all_data = self.site_data.get_data('all', ['hs', 'pca', 'chm', 'ndvi', 'mpsi'], 16, make_key=True)
        pass
    
    def init_dataloader(self):
        pass

    def init_model(self):
        pass

    def log_results(self):
        pass

    def run(self):
        pass
    



if __name__ == '__main__':
    #TODO: switch to argparse
    savedir = '/home/tony/thesis/lidar_hs_unsup_dl_model/experiment_logs'
    logfile = 'exp_logs.csv'
    datadir = '/home/tony/thesis/lidar_hs_unsup_dl_model/final_data'
    with open('experiments_test.csv') as csvfile:
        exp_reader = csv.DictReader(csvfile)
        for exp in exp_reader:
            new_exp = Experiment(**exp, savedir=savedir, logfile=logfile, datadir=datadir)
            Experiment.run()
