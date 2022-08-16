from sklearn.ensemble import RandomForestClassifier
from dataloaders import RenderedDataLoader
from torch.utils.data import DataLoader
import os
import torch
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


def get_rf_inputs(inp, features_dict):
    data = handle_inputs(inp, features_dict)
    crop_coords = inp['crop_coords']

    to_cat = []
    for ix, im in enumerate(data):
        to_append = im[...,crop_coords[0][ix]:crop_coords[0][ix]+ crop_coords[2][ix], crop_coords[1][ix]: crop_coords[1][ix] + crop_coords[3][ix]]
        to_cat.append(to_append.unsqueeze(0))
    data = torch.cat(to_cat, dim=0)
    data = data.numpy()
    data = rearrange(data, 'b c h w -> (b h w) c')

    targets = inp['targets']
    targets = torch.argmax(targets, dim=1).numpy()
    targets = rearrange(targets, 'b h w -> (b h w)')
    mask = data != data
    mask = reduce(mask, 'f c -> f', 'max')

    data = data[~mask]
    targets = targets[~mask]
    return data, targets

def rf_training(train_folder,
                valid_folder,
                test_folder,
                features_dict,
                pre_train_folder,
                rf_params={},
                ):
    train_dataset = RenderedDataLoader(train_folder, features_dict, stats_loc=os.path.join(pre_train_folder, 'stats/stats.npy'))
    train_loader = DataLoader(train_dataset, batch_size=len(train_dataset), num_workers=1)

    valid_dataset = RenderedDataLoader(valid_folder, features_dict, stats_loc=os.path.join(pre_train_folder, 'stats/stats.npy'))
    valid_loader = DataLoader(valid_dataset, batch_size=len(valid_dataset), num_workers=1)

    test_dataset = RenderedDataLoader(test_folder, features_dict, stats_loc=os.path.join(pre_train_folder, 'stats/stats.npy'))
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), num_workers=1)

    clf = RandomForestClassifier(**rf_params)

    for x in train_loader:
        train_data, train_targets = get_rf_inputs(x, features_dict)

    for x in test_loader:
        test_data, test_targets = get_rf_inputs(x, features_dict)
    

    clf.fit(train_data, train_targets)
    print(clf.score(test_data, test_targets))


    return None


if __name__ == '__main__':
    rf_training(
        pre_train_folder = 'C:/Users/tonyt/Documents/Research/datasets/tensors/niwo_2020_pca_blocks/raw_training',
        train_folder = 'C:/Users/tonyt/Documents/Research/datasets/tensors/niwo_2020_pca_blocks/sp_train',
        valid_folder = 'C:/Users/tonyt/Documents/Research/datasets/tensors/niwo_2020_pca_blocks/sp_valid',
        test_folder  = 'C:/Users/tonyt/Documents/Research/datasets/tensors/niwo_2020_pca_blocks/sp_test',
        features_dict={
            'pca': 16,
        }
    )