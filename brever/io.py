import os
import re
import random

import yaml
import numpy as np
import soundfile as sf
from resampy import resample
import sofa

from brever.config import defaults


def get_path(field_name):
    if os.path.exists('user_paths.yaml'):
        with open('user_paths.yaml') as f:
            user_paths = yaml.load(f)
        if user_paths[field_name] is not None:
            output = user_paths[field_name]
    else:
        config = defaults()
        output = getattr(config.PATH, field_name)
    if not os.path.exists(output):
        raise ValueError('the following dataset path was not found in the '
                         'filesystem: {output}')
    return output


def load_random_target(target_alias, lims=None, fs=16e3, randomizer=None):
    '''
    Load a random target signal from the target speech database.

    Parameters:
        target_alias:
            Target dataset alias. Can be either:
            - timit
            - libri
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
    if target_alias == 'timit':
        dirpath = get_path('TIMIT')
    elif target_alias == 'libri':
        dirpath = get_path('LIBRI')
    else:
        raise ValueError(f'wrong target alias: {target_alias}')
    if not os.path.exists(dirpath):
        raise ValueError(f'Directory not found: {dirpath}')
    all_filepaths = []
    for root, dirs, files in os.walk(dirpath):
        for file in files:
            if (file.endswith(('.wav', '.WAV')) and 'SA1' not in file and 'SA2'
                    not in file) or file.endswith('flac'):
                all_filepaths.append(os.path.join(root, file))
    if not all_filepaths:
        raise ValueError(f'No audio file found in {dirpath}')
    random.Random(0).shuffle(all_filepaths)
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


def load_brirs(room_alias, angles, fs=16e3):
    '''
    Load BRIRs from a given room alias and a list of angles.

    Parameters:
        room_alias:
            Room alias. Can be either:
            - 'surrey_anechoic'
            - 'surrey_room_a'
            - 'surrey_room_b'
            - 'surrey_room_c'
            - 'surrey_room_d'
            - 'huddersfield_c1m'
            - 'huddersfield_c2m'
            - 'huddersfield_c4m'
            - 'huddersfield_c6m'
            - 'huddersfield_c8m'
            - 'huddersfield_l1m'
            - 'huddersfield_l2m'
            - 'huddersfield_l4m'
            - 'huddersfield_l6m'
            - 'huddersfield_l8m'
            - 'huddersfield_lw1m'
            - 'huddersfield_lw2m'
            - 'huddersfield_lw4m'
            - 'huddersfield_lw6m'
            - 'huddersfield_lw8m'
        angles:
            Angles from which the BRIRs were recorded.
        fs:
            Sampling frequency to resample the BRIR to.

    Returns:
        brirs:
            List of BRIRs.
        fs:
            BRIRs sampling frequency.
    '''
    if angles is None:
        if room_alias.startswith('surrey_'):
            angles = np.linspace(-90, 90, 37)
        elif room_alias.startswith('huddersfield_'):
            angles = np.arange(100)/100*360
        else:
            raise ValueError(f'wrong room alias: {room_alias}')
        return load_brirs(room_alias, angles, fs)
    iterable = True
    try:
        iter(angles)
    except TypeError:
        iterable = False
    if iterable:
        brirs = []
        fss = []
        for angle in angles:
            brir, fs_ = load_brirs(room_alias, angle, fs)
            brirs.append(brir)
            fss.append(fs_)
        if not brirs:
            return [], None
        assert all(fs_ == fss[0] for fs_ in fss)
        return brirs, fss[0]
    angle = angles
    if room_alias.startswith('surrey_'):
        dirpath = get_path('SURREY')
        m = re.match('^surrey_(.*)$', room_alias)
        if m is None:
            raise ValueError(f'wrong room alias: {room_alias}')
        room_name = m.group(1)
        if room_name == 'anechoic':
            room_folder = 'Anechoic'
        else:
            m = re.match('^room_(.)$', room_name)
            if m is None:
                raise ValueError(f'wrong room alias: {room_alias}')
            room_letter = m.group(1)
            room_folder = 'Room_%s' % room_letter.upper()
        room_dir = os.path.join(dirpath, room_folder, '16kHz')
        r = re.compile('CortexBRIR_.*s_%ideg_16k.wav' % angle)
        filenames = list(filter(r.match, os.listdir(room_dir)))
        if len(filenames) > 1:
            raise ValueError('more than one brir was found for room '
                             f'{room_alias} and angle {angle} in filesystem')
        elif len(filenames) == 0:
            raise ValueError('could not find any brir for room '
                             f'{room_alias} and angle {angle} in filesystem')
        filepath = os.path.join(room_dir, filenames[0])
        brir, fs_ = sf.read(filepath)
    elif room_alias.startswith('huddersfield_'):
        dirpath = get_path('HUDDERSFIELD')
        m = re.match('^huddersfield_(.*)m$', room_alias)
        if m is None:
            raise ValueError(f'wrong room alias: {room_alias}')
        room_name = m.group(1)
        filename = f'{room_name.upper()}m.sofa'
        filepath = os.path.join(dirpath, 'Binaural', 'SOFA', filename)
        HRTF = sofa.Database.open(filepath)
        positions = HRTF.Source.Position.get_values(system='spherical')
        measurement = np.argwhere(abs(positions[:, 0] - angle) <= 1e-6)
        if len(measurement) == 0:
            raise ValueError('could not find any brir for room '
                             f'{room_alias} and angle {angle} in filesystem')
        elif len(measurement) > 1:
            raise ValueError('more than one brir was found for room '
                             f'{room_alias} and angle {angle} in filesystem')
        measurement = int(measurement)
        ir_l = HRTF.Data.IR.get_values({'M': measurement, 'R': 0})
        ir_r = HRTF.Data.IR.get_values({'M': measurement, 'R': 1})
        brir = np.vstack((ir_l, ir_r)).T
        fs_ = HRTF.Data.SamplingRate.get_values(indices={'M': measurement})
    else:
        raise ValueError(f'wrong room alias: {room_alias}')
    if fs_ != fs:
        brir = resample(brir, fs_, fs, axis=0)
    return brir, fs


def load_random_noise(noise_alias, n_samples, lims=None, fs=16e3,
                      randomizer=None):
    '''
    Load a random noise recording.

    Parameters:
        noise_alias:
            Noise alias. Can be either:
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
            Sampling frequency to resample the noise signal to.
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
    if noise_alias.startswith('dcase_'):
        dirpath = get_path('DCASE')
        all_filepaths = []
        m = re.match('^dcase_(.*)$', noise_alias)
        if m is None:
            raise ValueError('type_ should start with dcase_')
        prefix = m.group(1)
        for root, dirs, files in os.walk(dirpath):
            for file in files:
                if file.endswith(('.wav', '.WAV')) and file.startswith(prefix):
                    all_filepaths.append(os.path.join(root, file))
        if not all_filepaths:
            raise ValueError(f'No .wav file found in {dirpath}')
        random.Random(0).shuffle(all_filepaths)
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
            i_start = randomizer.randint(0, n_samples)
            indices = np.arange(n_samples) + i_start
            indices = indices % len(x)
            x = x[indices]
            i_end = None
        else:
            i_start = randomizer.randint(0, len(x) - n_samples)
            i_end = i_start+n_samples
            x = x[i_start:i_end]
        return x, filepath, (i_start, i_end)
    else:
        raise ValueError(f'wrong noise alias: {noise_alias}')
