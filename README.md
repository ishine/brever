# brever
Binaural speech segregation in noisy and reverberant environments using deep neural networks.

"brever" reads "reverb" backwards.

# Description

`brever` is a deep learning Python project implemented in PyTorch and that can be used for enhancement of speech corrupted by interfering noise and room reverberation. The package allows to generate datasets of noisy and reverberant mixtures from databases of clean speech, noise recordings and binaural room impulse responses (BRIRs). Great amounts of datasets and models can be created and trained thanks to an organization based on YAML configuration files and hash IDs to identify the models.

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
If on Windows run `venv/Scripts/activate.bat` to activate the environment instead.

3. Install requirements:
```
pip install -r requirements.txt
```
If on Windows, first install PyTorch manually as described in the [PyTorch](https://pytorch.org/) home webpage before installing the rest of the requirements.

4. Install the package in develop mode:
```
python setup.py develop
```

# External datasets

External databases of clean speech, noise recordings and binaural room impulse responses (BRIRs) are needed to generate the noisy speech mixture on which the models are trained. Currently, only the following databases are supported:

- Clean speech database:
  - [TIMIT](https://catalog.ldc.upenn.edu/LDC93S1)
  - [Libri](http://www.openslr.org/12)
- BRIR database:
  - [SURREY binaural room impulse responses (BRIR) database](https://ieeexplore.ieee.org/document/5473135)
- Noise database
  - [TAU Urban Acoustic Scenes 2019 development dataset](https://zenodo.org/record/2589280)

Place them e.g. under `data/external/TIMIT/`, `data/external/Libri/`, `data/external/SURREY/` and `data/external/DCASE/` respectively. Your directory tree should look like this:

```
brever
├── brever
│   └── ...
├── data
│   ├── external
│   │   ├── DCASE
│   │   │    └── ...
│   │   ├── TIMIT
│   │   │    └── ...
│   │   ├── SURREY
│   │   │    └── ...
│   │   └── TIMIT
│   │        └── ...
│   └── processed
│       └── ...
├── models
│   └── ...
├── scripts
│   └── ...
├── defaults.yaml
├── requirements.txt
├── setup.py
└── ...
```

# How to use

## Creating a dataset

To create a dataset, first create a new directory under `data/processed/`. Place inside that directory a `config.yaml` file with the preprocessing parameters you wish to overwrite. The complete list of preprocessing parameters can be found in `defaults.yaml`.

Then run `create_dataset.py` with the dataset directory as argument:

```
python scripts/create_dataset.py data/processed/my_dataset/
```

The following items are then created next to the YAML file:

- `dataset.hdf5`: the main dataset containing features and labels
- `mixture_info.json`: metadata about each simulated noisy mixture
- `pipes.pkl`: serialized objects used to preprocess the mixtures
- `log.txt`: a log file
- `peek.png`: a plot of a small amount of samples in the dataset
- `peek_standardized.png`: same but with features standardized
- `examples/`: a folder with example mixtures for verification

**Example**: If you wish to simulate 10 noisy mixtures, write the following `config.yaml` file:

```yaml
PRE:
  MIXTURES:
    NUMBER: 10
```

Save it under `data/processed/my_dataset/` such that your working tree looks like this:

```
brever
├── data
│   └── processed
│      └── my_dataset
│          └── config.yaml
└── ...
```

Then simply run:

```
python scripts/create_dataset.py data/processed/my_dataset/
```

Your working tree will then look like this:

```
brever
├── data
│   └── processed
│      └── my_dataset
│          ├── config.yaml
│          ├── dataset.hdf5
│          ├── mixture_info.json
│          ├── pipes.pkl
│          ├── log.txt
│          ├── peek.png
│          └── peek_standardized.png
│          └── examples
│              └── ...
└── ...
```

## Training a model

Similarly, to train a model, first create a `config.yaml` file with the post-processing and model parameters you wish to overwrite and place inside a new folder under the `models/` directory. Then run:

```
python scripts/train_model.py models/my_model/
```

**Example**: If you wish to train a model with the training dataset `data/processed/my_train_dataset` and the validation dataset `data/processed/my_val_dataset`, then create the following `config.yaml` file:

```yaml
POST:
  PATH:
    TRAIN: data/processed/training/
    VAL: data/processed/validation/
```

Save it under `models/my_model/` such that your working tree looks like this:

```
brever
├── models
│   └── my_model
│       └── config.yaml
└── ...
```
Then simply run:

```
python scripts/train_model.py models/my_model/
```

## Creating multiple models

Models can be created programmatically using the `scripts\initialize_models.py` script. For example, if you want to initialize models that use MFCC or IC features, and with 1 or 2 hidden layers, then you can call:

```
python scripts/initialize_models.py --features mfcc pdf --layers 1 2
```

This will initialize 4 models (since we have a 2x2 grid of parameters) under the `models/` directory. For each model, a new directory called after a unique hash ID is created under the `models/` directory, and the corresponding `config.yaml` is saved inside that directory.

This allows to quickly initialize great amounts of models while having control over the parameters we wish to investigate. The complete list of initializable parameters can be displayed using `python scripts/initialize_models.py -h`.

Most of the `brever` scripts accept glob patterns. This means that in order to train all the available models, one can simply run:

```
python scripts/train_model.py models/*
```

## Creating testing datasets

Unlike the training and validation datasets, the testing datasets must follow a naming convention according to varying signal-to-noise ratio (SNR) and room. To initialize the testing datasets, one must run:

```
python scripts/initialize_testing_datasets.py alias
```

Where `alias` is a short tag of choice that will be used to name of the testing datasets. This will create the following folders in the `data/processed/` directory:

```
brever
├── data
│   └── processed
│      └── testing_alias_snr0_roomA
│      └── testing_alias_snr0_roomB
│      └── testing_alias_snr0_roomC
│      └── testing_alias_snr0_roomD
│      └── testing_alias_snr3_roomA
│      └── testing_alias_snr3_roomB
│      └── testing_alias_snr3_roomC
│      └── testing_alias_snr3_roomD
│      └── ...
│      └── testing_alias_snr15_roomA
│      └── testing_alias_snr15_roomB
│      └── testing_alias_snr15_roomC
│      └── testing_alias_snr15_roomD
└── ...
```

The datasets should be then created using the `scripts/create_dataset.py` script:

```
python scripts/create_dataset.py data/processed/testing_alias_snr*
```

## Testing a model

To test a freshly trained model, the `POST.PATH.TEST` field in the `config.yaml` file of the model should point to the start of the naming convention of the testing datasets. For example, to train a model on the testing datasets listed above, the `config.yaml` file of the model should contain this:

```yaml
POST:
  PATH:
    TEST: data/processed/testing_alias
```

The testing script will then look for the testing datasets by appending `_snrX_roomY/` to that path.

Once this is set, the model is tested my simply calling:

```
python scripts/test_model.py model/my_model/
```

This will create a series of new files in the model directory:
- The MSE scores are stored in `mse_scores.npy`
- The PESQ scores are stored in `pesq_scores.mat`
- The segmental scores (segSSNR, segBR, segNR and segRR) are stored in `seg_scores.npy`

In each file, the scores are arranged by SNR and room for further analysis.

