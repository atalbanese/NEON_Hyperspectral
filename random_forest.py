from sklearn.ensemble import RandomForestClassifier
import sklearn.metrics as sm
from dataloaders import RenderedDataLoader
from torch.utils.data import DataLoader
import os
import torch
import numpy as np
from einops import rearrange, reduce

def handle_inputs(inp, features_dict):
    to_cat = []
    for key, value in features_dict.items():
        if key == 'mask':
            continue
        if key == 'targets':
            continue
        if key == 'height':
            continue
        cur = inp[key]
        if len(cur.shape) != 4:
            cur = cur.unsqueeze(1)
        if cur.shape[1] != value:
            cur = cur[:,:value,...]
        to_cat.append(cur)
    combined = torch.cat(to_cat, dim=1)

    return combined


def get_rf_inputs(inp, features_dict, filters):
    data = handle_inputs(inp, features_dict)
    filter_masks = []
    for f, t in filters.items():
        values = inp['mask'][f]
        values = values.numpy()
        mask = values < t
        mask = rearrange(mask, 'b h w -> (b h w)')
        filter_masks.append(mask)

    data = data.numpy()
    data = rearrange(data, 'b c h w -> (b h w) c')
    missing_mask = data != data
    missing_mask = reduce(missing_mask, 'f c -> f', 'max')

    targets = inp['targets']
    targets = targets.numpy()
    targets = rearrange(targets, 'b h w -> (b h w)')
    targets_mask = targets == -1

    if len(filter_masks) > 0:
        for mask in filter_masks:
            targets_mask = targets_mask + mask
    final_mask = targets_mask + missing_mask

    

    data = data[~final_mask]
    targets = targets[~final_mask]

    return data, targets

def pixel_split_inputs(data, targets, train_prop=0.7):
    rng = np.random.default_rng()
    data_length = data.shape[0]
    sample_range = list(range(data_length))

    num_samples = int(data_length * train_prop)

    train_samples = rng.choice(sample_range, size =num_samples, replace=False)

    test_samples = list(set(sample_range) - set(train_samples))

    train_data = data[train_samples]
    train_targets = targets[train_samples]

    test_data = data[test_samples]
    test_targets = targets[test_samples]
    # print(train_data.shape)
    # print(train_targets.shape)
    # print(test_data.shape)


    return train_data, train_targets, test_data, test_targets





def rf_training(train_folder='',
                test_folder='',
                features_dict={},
                pre_train_folder='',
                rf_params={},
                pixel_split=False,
                filters={}
                ):
    train_dataset = RenderedDataLoader(train_folder, features_dict, stats_loc=os.path.join(pre_train_folder, 'stats/stats.npy'), scaling=False)
    train_loader = DataLoader(train_dataset, batch_size=len(train_dataset), num_workers=1)

    # valid_dataset = RenderedDataLoader(valid_folder, features_dict, stats_loc=os.path.join(pre_train_folder, 'stats/stats.npy'), scaling=False)
    # valid_loader = DataLoader(valid_dataset, batch_size=len(valid_dataset), num_workers=1)

    

    clf = RandomForestClassifier(**rf_params)

    for x in train_loader:
        train_data, train_targets = get_rf_inputs(x, features_dict, filters)

    
    
    if not pixel_split:
        test_dataset = RenderedDataLoader(test_folder, features_dict, stats_loc=os.path.join(pre_train_folder, 'stats/stats.npy'), scaling=False)
        test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), num_workers=1)

        for x in test_loader:
            test_data, test_targets = get_rf_inputs(x, features_dict, filters)

    else:
        train_data, train_targets, test_data, test_targets = pixel_split_inputs(train_data, train_targets, train_prop=0.7)        

    clf.fit(train_data, train_targets)
    predictions = clf.predict(test_data)
    conf_matrix = sm.confusion_matrix(test_targets, predictions)
    print(conf_matrix)
    print(clf.score(test_data, test_targets))




    return None


if __name__ == '__main__':
    rf_training(
        pre_train_folder = 'C:/Users/tonyt/Documents/Research/datasets/tensors/niwo_2020_pca_blocks/raw_training',
        train_folder = 'C:/Users/tonyt/Documents/Research/datasets/tensors/rf_test/scholl/pca_all',
        pixel_split=True,
        #test_folder  = 'C:/Users/tonyt/Documents/Research/datasets/tensors/rf_test/pca_object_split/test',
        features_dict={
            'pca': 16,
        },
        filters={
            'shadow': 0.03,
            'ndvi': 0.2
        }
    )

    rf_training(
        pre_train_folder = 'C:/Users/tonyt/Documents/Research/datasets/tensors/niwo_2020_pca_blocks/raw_training',
        train_folder = 'C:/Users/tonyt/Documents/Research/datasets/tensors/rf_test/scholl/pca_object_split/train',
        pixel_split=False,
        test_folder  = 'C:/Users/tonyt/Documents/Research/datasets/tensors/rf_test/scholl/pca_object_split/test',
        features_dict={
            'pca': 16,
        },
        filters={
            'shadow': 0.03,
            'ndvi': 0.2
        }
    )
    rf_training(
        pre_train_folder = 'C:/Users/tonyt/Documents/Research/datasets/tensors/niwo_2020_pca_blocks/raw_training',
        train_folder = 'C:/Users/tonyt/Documents/Research/datasets/tensors/rf_test/scholl/pca_plot_split/train',
        pixel_split=False,
        test_folder  = 'C:/Users/tonyt/Documents/Research/datasets/tensors/rf_test/scholl/pca_plot_split/test',
        features_dict={
            'pca': 16,
        },
        filters={
            'shadow': 0.03,
            'ndvi': 0.2
        }
    )