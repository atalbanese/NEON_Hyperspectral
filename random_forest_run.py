from splitting import SiteData
from torch_data import PaddedTreeDataSet
from sklearn.ensemble import RandomForestClassifier
import sklearn.metrics as sm
from torch.utils.data import DataLoader
from torch_pretraining_model import PreTrainingModel
from sklearn.decomposition import PCA


if __name__ == "__main__":
    niwo = SiteData(
        site_dir = '/home/tony/thesis/data/NIWO',
        random_seed=42,
        train = 0.6,
        test= 0.3,
        valid = 0.1)

    niwo.make_splits('plot_level')
    train_data = niwo.get_data("training and validation", ['hs', 'origin'], 16, make_key=True)
    test_data = niwo.get_data('testing', ['hs', 'origin'], 16, make_key=True)

    train_set = PaddedTreeDataSet(train_data, pad_length=16, stats='/home/tony/thesis/data/stats/niwo_stats.npz', augments_list=["normalize"])
    train_loader = DataLoader(train_set, batch_size=len(train_set), num_workers=0)

    test_set = PaddedTreeDataSet(test_data, pad_length=16, stats='/home/tony/thesis/data/stats/niwo_stats.npz', augments_list=["normalize"])
    test_loader = DataLoader(test_set, batch_size=len(test_set), num_workers=0)

    for x in train_loader:
        pass
    for y in test_loader:
        pass

    # pt_model = PreTrainingModel.load_from_checkpoint('/home/tony/thesis/pre_training_ckpts/pre_training_1.ckpt')
    # enhanced = pt_model(x)

    # inp = enhanced[~x['hs_pad_mask']].numpy()
    # enhanced_y = pt_model(y)

    # test_inp =enhanced_y[~y['hs_pad_mask']].numpy()

    test_inp = y['hs'][~y['hs_pad_mask']].numpy()
    inp = x['hs'][~x['hs_pad_mask']].numpy()

    # pca = PCA(n_components=4, svd_solver='full')

    # inp = pca.fit_transform(inp)
    # test_inp = pca.transform(test_inp)
    targets = x['single_target'][~x['hs_pad_mask']].numpy()

    
    
    test_targets = y['single_target'][~y['hs_pad_mask']].numpy()
    clf = RandomForestClassifier()

    clf.fit(inp, targets)
    predictions = clf.predict(test_inp)
    conf_matrix = sm.confusion_matrix(test_targets, predictions)
    print(conf_matrix)
    print(clf.score(test_inp, test_targets))


