# brever
Binaural speech segregation in noisy and reverberant environments using deep neural networks.

"brever" reads "reverb" backwards.

# Description

TODO

# Installation

## Requirements

TODO

## Getting the external datasets

Obtain the [TIMIT speech database](https://catalog.ldc.upenn.edu/LDC93S1), the [SURREY binaural room impulse responses (BRIR) database](https://ieeexplore.ieee.org/document/5473135) and the [TAU Urban Acoustic Scenes 2019 development dataset](https://zenodo.org/record/2589280) and place them under `data/external/TIMIT/`, `data/external/SURREY/` and `data/external/DCASE/` respectively. Your directory tree should look like this:

```
brever
├── brever
│   └── ...
├── notebooks
│   └── ...
├── data
│   ├── external
│   │   ├── DCASE
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
└── README.md
```

# How to use

## Creating a dataset

To create a dataset, first create a new directory under `data/processed/`. Then write a YAML file with the preprocessing parameters you wish to overwrite and save it under the directory you just created. The complete list of preprocessing parameters can be found in `brever/config.py`.

Then run `create_dataset.py` with the dataset directory as argument:

```
python scripts/create_dataset.py data/processed/dataset_name
```

The following files are then created next to the YAML file:

- `dataset.hdf5`: the main dataset containing features and labels
- `mixture_info.json`: metadata about each simulated noisy mixture
- `pipes.pkl`: serialized objects used to preprocess the mixtures
- `log.txt`: a log file
- `peek.png`: a plot of a small amount of samples in the dataset
- `peek_standardized.png`: same but with features standardized

**Example**: If you wish to simulate 10 noisy mixtures, write the following `config.yaml` file:

```yaml
PRE:
  MIXTURES:
    NUMBER: 10
```

Create a directory e.g. `my_dataset/` under `data/processed/` and save the YAML file under that directory such that your working tree looks like this:

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
python scripts/create_dataset.py data/processed/dataset_name
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
└── ...
```

## Training a model

TODO

## Testing a model

TODO
