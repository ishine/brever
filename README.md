# Brever
Binaural speech segregation in noisy and reverberant environments using deep neural networks.

What can you do with Brever?
* Generate datasets of noisy and reverberant mixtures from a range of supported databases of speech utterances, noise recordings and binaural room impulse responses (BRIRs)
* Train PyTorch-based neural networks to perform speech enhancement. Currently implemented models are
** Feed-forward neural network (FFNN)
** Conv-TasNet
* Evaluate trained models in terms of different metrics: SNR, PESQ and STOI improvements.

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

External databases of speech utterances, noise recordings and binaural room impulse responses (BRIRs) are required to generate datasets of noisy and reverberant mixtures. The following databases are currently supported:

- Speech databases:
  - [TIMIT](https://doi.org/10.35111/17gk-bn40)
  - [LibriSpeech](http://www.openslr.org/12)
  - ~~HINT~~
  - ~~IEEE~~
  - ~~ARCTIC~~
  - [WSJ](https://doi.org/10.35111/ewkm-cg47)
  - [VCTK](https://doi.org/10.7488/ds/2645)
  - [Clarity](https://doi.org/10.17866/rd.salford.16918180.v3)
- BRIR databases:
  - [Surrey](https://doi.org/10.1109/TASL.2010.2051354)
  - [ASH](https://github.com/ShanonPearce/ASH-IR-Dataset)
  - [BRAS](https://doi.org/10.1016/j.apacoust.2020.107867)
  - [CATT](http://iosr.surrey.ac.uk/software/index.php#CATT_RIRs)
  - [AVIL](https://doi.org/10.17743/jaes.2020.0026)
- Noise databases:
  - [TAU](https://doi.org/10.5281/zenodo.2589280)
  - [NOISEX](https://doi.org/10.1016/0167-6393(93)90095-3)
  - [ICRA](https://pubmed.ncbi.nlm.nih.gov/11465297/)
  - [DEMAND](https://doi.org/10.5281/zenodo.1227121)
  - [ARTE](https://doi.org/10.5281/zenodo.2261633)

These should be placed in the paths specified in `config/paths.yaml`.

# How to use

## Creating datasets

You can initialize a dataset using `scripts/init_dataset.py`. The script takes as optional arguments the parameters for the dataset. It creates a new directory under `data/datasets/train/` or `data/datasets/test/` depending on which mandatory argument was used between `train` or `test`. The new directory contains a `config.yaml` file with all the dataset parameters. The directory is named after a unique hash ID calculated from the `config.yaml` file. This is to prevent initializing duplicate datasets.

The dataset is then created using the `scripts/create_dataset.py` script:

```
$ python scripts/init_dataset.py train
Initialized data/datasets/train/5818d1fb/config.yaml
$ python scripts/create_dataset.py data/datasets/train/5818d1fb/
```

The following files are then created next to the `config.yaml` file:

- `audio.tar`: an archive containing audio files of noisy and reverberant speech mixtures
- `log.log`: a log file
- `mixture_info.json`: metadata about each mixture

## Training models

You can initialize a model using `scripts/init_model.py`. The script takes as optional arguments the training parameters, and as a mandatory sub-command the model architecture. The sub-command then takes as optional arguments the parameters for the model. The script creates a new directory under `models/` containing a `config.yaml` file with all the model parameters. The directory is named after a unique hash ID calculated from the `config.yaml` file. This is to prevent initializing duplicate models.

The model is then trained using the `scripts/train_model.py` script:

```
$ python scripts/init_model.py --train-path data/datasets/train/5818d1fb/ convtasnet
Initialized models/ece4a25b/config.yaml
$ python scripts/train_model.py models/ece4a25b/
```

The following files are then created next to the `config.yaml` file:

- `checkpoint.pt`: the model PyTorch state dictionary
- `log.txt`: a log file
- `losses.npz`: training and validation curves in NumPy format
- `training_curve.png`: a plot of the training and validation curves

## Testing models

You can evaluate a trained model with `scripts/test_model.py`:

```
python scripts/test_model.py models/ece4a25b/ data/datasets/<dataset_id>/
```

This creates a `scores.json` containing the SNR, PESQ and STOI scores for the enhanced mixtures from the model as well as the unprocessed input mixtures.
