import os
import re
import random

import numpy as np
import soundfile as sf
from resampy import resample


def load_random_target(target_dirpath, lims=None, fs=16e3, randomizer=None):
    '''
    Load a random target signal from the target speech database.

    Parameters:
        target_dirpath:
            Path to the target speech database in the filesystem (TIMIT or
            LibriSpeech).
        lims:
            Lower and upper fraction of files of the total list of files from
            which to randomly chose a target from. E.g. setting lims to
            (0, 0.5) means the first half of the files will be sampled. Can be
            left as None to chose from all the files.
        fs:
            Sampling rating to resample the signal to. If it matches the
            original sampling rate, the signal is returned as is.
        randomizer:
            Custom random.Random instance. Useful for seeding.

    Returns:
        x:
            Audio signal.
        filepath:
            Path to the loaded file.
    '''
    if not os.path.exists(target_dirpath):
        raise ValueError(f'Directory not found: {target_dirpath}')
    all_filepaths = []
    for root, dirs, files in os.walk(target_dirpath):
        for file in files:
            if (file.endswith(('.wav', '.WAV')) and 'SA1' not in file and 'SA2'
                    not in file) or file.endswith('flac'):
                all_filepaths.append(os.path.join(root, file))
    if not all_filepaths:
        raise ValueError(f'No audio file found in {target_dirpath}')
    state = random.getstate()
    random.seed(42)
    random.shuffle(all_filepaths)
    random.setstate(state)
    if lims is not None:
        n_files = len(all_filepaths)
        i_min, i_max = round(n_files*lims[0]), round(n_files*lims[1])
        all_filepaths = all_filepaths[i_min:i_max]
    if randomizer is None:
        randomizer = random
    filepath = randomizer.choice(all_filepaths)
    x, fs_old = sf.read(filepath)
    if fs_old != fs:
        x = resample(x, fs_old, fs, axis=0)
    return x, filepath


def load_brir(surrey_dirpath, room_alias, angle):
    '''
    Load a BRIR from the SURREY database given a room alias and a BRIR angle.

    Parameters:
        surrey_dirpath:
            Path to the SURREY database in the filesystem.
        room_alias:
            Room alias. Can be either:
            - 'surrey_anechoic'
            - 'surrey_room_a'
            - 'surrey_room_b'
            - 'surrey_room_c'
            - 'surrey_room_d'
        angle:
            Angle from which the BRIR was recorded with. For SURREY, the
            possible angles range from -90 to 90 with a step of 5.

    Returns:
        brir:
            BRIR.
        fs:
            BRIR sampling frequency.
    '''
    m = re.match('^surrey_(.*)$', room_alias)
    if m is None:
        raise ValueError('could not find room %s in filesystem' % room_alias)
    room_name = m.group(1)
    if room_name == 'anechoic':
        room_folder = 'Anechoic'
    else:
        m = re.match('^room_(.)$', room_name)
        if m is None:
            raise ValueError(('could not find room %s in filesystem'
                              % room_alias))
        room_letter = m.group(1)
        room_folder = 'Room_%s' % room_letter.upper()
    room_dir = os.path.join(surrey_dirpath, room_folder, '16kHz')
    r = re.compile('CortexBRIR_.*s_%ideg_16k.wav' % angle)
    filenames = list(filter(r.match, os.listdir(room_dir)))
    if len(filenames) > 1:
        raise ValueError('more than one file matches pattern')
    elif len(filenames) == 0:
        raise ValueError(('could not find room %s with angle %i in '
                          'filesystem' % room_alias))
    filepath = os.path.join(room_dir, filenames[0])
    return sf.read(filepath)


def load_brirs(surrey_dirpath, room_alias, angles=None):
    '''
    Load multiple BRIRs from the SURREY database given a room alias and a list
    of angles.

    Parameters:
        surrey_dirpath:
            Path to the SURREY database in the filesystem.
        room_alias:
            Room alias. Can be either:
            - 'surrey_anechoic'
            - 'surrey_room_a'
            - 'surrey_room_b'
            - 'surrey_room_c'
            - 'surrey_room_d'
        angles:
            List of angles from which the BRIR were recorded with. For SURREY,
            the possible angles range from -90 to 90 with a step of 5.

    Returns:
        brirs:
            List of BRIRs. Same length as angles.
        fs:
            Common sampling frequency of the BRIRs.
    '''
    brirs = []
    fss = []
    if angles is None:
        m = re.match('^surrey_(.*)$', room_alias)
        if m is None:
            raise ValueError(('could not find room %s in filesystem'
                              % room_alias))
        room_name = m.group(1)
        if room_name == 'anechoic':
            room_folder = 'Anechoic'
        else:
            m = re.match('^room_(.)$', room_name)
            if m is None:
                raise ValueError(('could not find room %s in filesystem'
                                  % room_alias))
            room_letter = m.group(1)
            room_folder = 'Room_%s' % room_letter.upper()
        room_dir = os.path.join(surrey_dirpath, room_folder, '16kHz')
        for filename in os.listdir(room_dir):
            brir, fs = sf.read(os.path.join(room_dir, filename))
            brirs.append(brir)
            fss.append(fs)
    elif not angles:
        return [], None
    else:
        for angle in angles:
            brir, fs = load_brir(surrey_dirpath, room_alias, angle)
            brirs.append(brir)
            fss.append(fs)
    if any(fs != fss[0] for fs in fss):
        raise ValueError('the brirs do not all have the same samplerate')
    return brirs, fss[0]


def load_random_noise(dcase_dirpath, type_, n_samples, lims=None, fs=16e3,
                      randomizer=None):
    '''
    Load a random noise recording from the DCASE Challenge 2019 Task 1
    development set.

    Parameters:
        dcase_dirpath:
            Path to the DCASE Challenge 2019 Task 1 development set in the
            filesystem
        type_:
            Noise type. Can be either:
            - 'dcase_airport'
            - 'dcase_bus'
            - 'dcase_metro'
            - 'dcase_park'
            - 'dcase_public_square'
            - 'dcase_shopping_mall'
            - 'dcase_street_pedestrian'
            - 'dcase_street_traffic'
            - 'dcase_tram'
        n_samples:
            Number of samples to load.
        lims:
            Lower and upper fraction of files of the total list of files from
            which to randomly chose a sample from. E.g. setting lims to
            (0, 0.5) means the first half of the files will be sampled. Can be
            left as None to chose from all the files.
        fs:
            Sampling frequency to which resample the noise signal to.
        randomizer:
            Custom random.Random instance. Useful for seeding.

    Returns:
        x:
            Noise signal.
        filepath:
            Noise filepath in filesystem.
        indices:
            Starting and ending indices of the audio sample extracted from
            filepath.
    '''
    if not os.path.exists(dcase_dirpath):
        raise ValueError(f'Directory not found: {dcase_dirpath}')
    all_filepaths = []
    m = re.match('^dcase_(.*)$', type_)
    if m is None:
        raise ValueError('type_ should start with dcase_')
    prefix = m.group(1)
    for root, dirs, files in os.walk(dcase_dirpath):
        for file in files:
            if file.endswith(('.wav', '.WAV')) and file.startswith(prefix):
                all_filepaths.append(os.path.join(root, file))
    if not all_filepaths:
        raise ValueError(f'No .wav file found in {dcase_dirpath}')
    state = random.getstate()
    random.seed(42)
    random.shuffle(all_filepaths)
    random.setstate(state)
    if lims is not None:
        n_files = len(all_filepaths)
        i_min, i_max = round(lims[0]*n_files), round(lims[1]*n_files)
        all_filepaths = all_filepaths[i_min:i_max]
    if randomizer is None:
        randomizer = random
    filepath = randomizer.choice(all_filepaths)
    x, fs_old = sf.read(filepath)
    if x.ndim == 2:
        x = x[:, 0]
    if fs_old != fs:
        x = resample(x, fs_old, fs)
    if len(x) < n_samples:
        i_start = random.randint(0, n_samples)
        indices = np.arange(n_samples) + i_start
        indices = indices % len(x)
        x = x[indices]
        i_end = None
    else:
        i_start = random.randint(0, len(x) - n_samples)
        i_end = i_start+n_samples
        x = x[i_start:i_end]

    return x, filepath, (i_start, i_end)
