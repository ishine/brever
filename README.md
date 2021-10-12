# Brever
Binaural speech segregation in noisy and reverberant environments using deep neural networks.

Brever is a speech enhancement package based on PyTorch. It allows to generate datasets of noisy and reverberant mixtures from databases of clean speech, noise recordings and binaural room impulse responses (BRIRs). Models can then be trained to enhance speech and evaluated using different metrics.

"Brever" reads "reverb" backwards.

# Installation

1. Clone the repo and `cd` to it:
```
git clone https://github.com/philgzl/brever.git
cd brever
```

2. Set up a virtual environment and activate it:
```
python -m venv venv
source venv/bin/activate
```

3. Install requirements:
```
pip install -r requirements.txt
```

4. Install the package in development mode:
```
python setup.py develop
```

# External databases

External databases of clean speech, noise recordings and binaural room impulse responses (BRIRs) are required to generate the noisy speech mixtures on which the models are trained. Currently, the following databases are supported:

- Speech databases:
  - [TIMIT](https://catalog.ldc.upenn.edu/LDC93S1)
  - [LibrisSpeech](http://www.openslr.org/12)
  - HINT
  - IEEE
  - ARCTIC
- BRIR databases:
  - [Surrey](https://ieeexplore.ieee.org/document/5473135)
  - ASH
  - BRAS
  - CATT
  - AVIL
- Noise databases:
  - [TAU Urban Acoustic Scenes 2019 development dataset](https://zenodo.org/record/2589280)
  - NOISEX
  - ICRA
  - DEMAND
  - ARTE

By default, these should be placed under `data/external/` as shown below. Alternatively you can change the paths in `defaults.yaml`, or create a `defaults_user.yaml` file in the root directory and overwrite the paths.

```
brever/
├── brever/
├── data/
│   ├── external/
│   │   ├── TIMIT/
│   │   ├── LibriSpeech/
│   │   ├── HINT/
│   │   ├── IEEE/
│   │   ├── ARCTIC/
│   │   ├── Surrey/
│   │   ├── ASH/
│   │   ├── BRAS/
│   │   ├── CATT/
│   │   ├── AVIL/
│   │   ├── DCASE/
│   │   ├── NOISEX/
│   │   ├── ICRA/
│   │   ├── DEMAND/
│   │   └── ARTE/
│   └── processed/
├── models/
├── scripts/
├── defaults.yaml
├── README.md
├── requirements.txt
└── setup.py
```

# How to use

## Configuration files

Datasets and models are initialized by creating a `config.yaml` file with the parameters to overwrite. The default parameters are defined in `defaults.yaml` in the root directory. User default parameters can also be defined by creating a `defaults_user.yaml` file in the root directory. When creating a dataset or training/testing a model, the corresponding scripts first read `defaults.yaml`, then `defaults_user.yaml` if it exists, and finally the `config.yaml` of the input dataset or model.

Datasets should be placed under `data/processed/train/`, `data/processed/val/` or `data/processed/test/`. Models should be placed under `models/`.

## Creating a dataset

You can initialize a dataset programmatically using `scripts/initialize_dataset.py`. This creates a new directory under `data/processed/train/`, `data/processed/val/` or `data/processed/test/` depending on which option was used between `--train`, `--val` and`--test`. The new directory contains the `config.yaml` file.

If no `--name` argument is provided, the directory is named after a unique hash ID calculated from the `config.yaml` file. This can be handy to prevent initializing and creating duplicate datasets.

You can then create the dataset with `scripts/create_dataset.py`:

```
python scripts/initialize_dataset.py --train --name my_dataset
python scripts/create_dataset.py data/processed/my_dataset/
```

The following are then created in the dataset directory:

- `dataset.hdf5`: the main dataset file containing features and labels
- `mixture_info.json`: metadata about each mixture
- `pipes.pkl`: serialized objects used to create the mixtures
- `log.txt`: a log file
- `peek.png`: a plot of a small amount of observations in the dataset
- `peek_standardized.png`: same but with standardized features
- `examples/`: a folder with example mixtures for verification

## Training a model

You can initialize a model programmatically using `scripts/initialize_models.py`. This creates a new directory under `models/` containing the `config.yaml` file. The directory is named after a unique hash ID calculated from the `config.yaml` file. No custom name can be provided.

Multiple models can be initialized simultaneously. The options in `scripts/initialize_models.py` can take multiple values and the entire grid of models is proposed to be intialized.

You can then train the model with `scripts/train_model.py`:

```
python scripts/initialize_models.py \
    --train-path data/processed/my_train_dataset/ \
    --val-path data/processed/my_val_dataset/ \
    --test-path data/processed/my_test_dataset/
python scripts/train_model.py models/my_model/
```

The following are then created in the model directory:

- `checkpoint.pt`: model state dictionary
- `config_full.yaml`: a complete configuration file including the parameters in `defaults.yaml` and `defaults_user.yaml` at the time of training
- `log.txt`: a log file
- `losses.npz`: training and validation curves in NumPy format
- `model_args.yaml`: the arguments used to initialize the model
- `training.png`: a plot of the training and validation curves
- `statistics.npy`: normalization statistics of the training dataset

## Testing a model

You can evaluate a trained model with `scripts/test_model.py`:

```
python scripts/test_model.py model/my_model/
```

This creates a `scores.json` file with the MSE, PESQ and STOI scores for the model as well as the PESQ and STOI scores for the unprocessed mixtures.

## Utility scripts

Initializing multiple datasets and models can quickly become overwhelming due to the assigned hash IDs, which is when the following scripts can be useful:
- `scripts/find_model.py` can scan for models in the `models/` directory
- `scripts/find_dataset.py` can scan for datasets in the `data/processed/` directory
- `scripts/check_sanity.py` checks that the models under `models/` are named after the correct hash ID, among other things.
