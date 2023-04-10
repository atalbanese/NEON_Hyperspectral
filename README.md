# NEON_Hyperspectral
Scripts to train deep learning transfomer and random forest tree species classification models using hyperspectral and vegetation data from NEON science sites. All tools required to download and annotate data are included. Developed as part of my master's thesis in GeoInformatics at Hunter College, 2023.

The general workflow is:

Download data -> Compress data with PCA -> Annotate data ->  Train and evaluate model

acquire_all_data.R -> site_pca.py -> annotate.py -> experiment_runner.py

## Download data

Rscript acquire_all_data.R -s SITENAME -d DATA_DIRECTORY -y YEAR

Downloading data for Rocky Mountain National Park in 2020. Will be saved in ...final_data/RMNP

```Rscript acquire_all_data.R -s RMNP -d C:/Users/tonyt/Documents/Research/final_data -y 2020```

## Create PCA Images from Hyperspectral Data
python site_pca.py SITENAME DATA_DIRECTORY

```python site_pca.py RMNP C:/Users/tonyt/Documents/Research/final_data```

The --alternate option can be used to fit a PCA model to one site and then compress a different site

```
# Fit PCA to RMNP and then compress NIWO images
python site_pca.py RMNP C:/Users/tonyt/Documents/Research/final_data --alternate NIWO
```

## Annotate data using different tree selection algorithms
```
python annotation.py RMNP C:\Users\tonyt\Documents\Research\final_data EPSG:32613 filtering -a
python annotation.py RMNP C:\Users\tonyt\Documents\Research\final_data EPSG:32613 snapping -a
python annotation.py RMNP C:\Users\tonyt\Documents\Research\final_data EPSG:32613 scholl -a
```

EPSG codes are not automatically generated and must be provided manually by the user. NEON sites are all UTM codes. 

Pixels can also be manually included/excluded using a GUI with the -m flag. Useful if you want to hand-select for sunlit pixels.
```
python annotation.py RMNP C:\Users\tonyt\Documents\Research\final_data EPSG:32613 filtering -m
```

## Train and evaluate models using experiments listed in a CSV file
Note: While this project works on both Linux and Windows, model training is much faster on Linux.


python experiment_runner.py LOG_DIR RESULTS_FILE DATA_DIRECTORY EXPERIMENTS.CSV

```python experiment_runner.py /home/tony/thesis/lidar_hs_unsup_dl_model/experiment_logs/ experiment_set_1_results.csv /home/tony/thesis/lidar_hs_unsup_dl_model/final_data/ experiments_set_1.csv```

### CSV Format

| exp_number  | sitename | anno_method | man_or_auto | split_method | model | apply_filters | inp_key | num_trials | remove_taxa | test_site | pt_ckpt |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| 1 | RMNP | filtering | auto | plot | DL | F | pca | 5 | PIFL2 | NIWO | pre_training_RMNP.ckpt |

#### CSV Key

| Header Name | Description | Options |
| --- | --- | --- |
| exp_number | Experiment Number, used for tracking experiments | Any integer |
| sitename | Acronym for NEON Site, ie RMNP | Any four character NEON sitecode |
| anno_method | Annotation method used to create training data | filtering, snapping, scholl |
| man_or_auto | Were pixels automatically or manually annotated? | man, auto |
| split_method | How should training, testing, and validation data be split | tree, plot, pixel |
| model | What kind of model to train and evaluate | DL, RF |
| apply_filters | Whether to apply vegetation and shadow filters to data | T, F |
| inp_key | What input from training data to train model on | pca, hs |
| num_trials | How many times to run this experiment | Any positive integer |
| remove_taxa | A list of taxa using NEON taxa codes to omit from training and evaluation, separated by semicolons. Leave blank to keep all | i.e. PIFL2;PICOL |
| test_site | NEON site to test trained model on. Leave blank to test on same site. Used to train on one site and test on another which has overlapping taxa| Any four character NEON sitecode |
| pt_ckpt | Model checkpoint to load trained on unlabelled data. Only usable with deep learning. Generated using pretraining_run.py. Can be left blank. | A file location ending in .ckpt |

## Pre-Training
Optionally, users may train and then supply a deep learning model pre-trained on unlabelled NEON site data using a SwAV (Swapping Assignments between Views) model (https://arxiv.org/abs/2006.09882)

python pretraining_run.py SAVEDIR DATA_DIRECTORY SITENAME

## Inference

Use a trained model to run inference on scenes. Coming soon!

## Requirements

Full requirements incoming, but general requirements are:

- R:
  - sp
  - sf
  - raster
  - neonUtilities
  - geoNEON
  - neonOS
  - tidyverse
  - lidR
  
- Python:
  - sklearn
  - geopandas
  - rasterio
  - pytorch
  - einops
  - pytorch-lightning
  - numpy
  - scikit-image
  - h5py
