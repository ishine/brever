import os
import re
import soundfile as sf
import random
from resampy import resample

from . import config


def load_random_target(lims=None, fs=16e3):
    '''
    Load a random target signal. Currently a random sentence from the TIMIT
    database is loaded. See config.py to set the path to the TIMIT database
    in your filesystem.

    Parameters:
        lims:
            Lower and upper fraction of files of the total list of files from
            which to randomly chose a target from. E.g. setting lims to
            (0, 0.5) means the first half of the files will be sampled. Can be
            left as None to chose from all the files.
        fs:
            Sampling rating to resample the signal to. If it matches the
            original sampling rate, the signal is returned as is.

    Returns:
        x:
            Audio signal.
        filepath:
            Path to the loaded file.
    '''
    dirpath = config.TIMIT_PATH
    all_filepaths = []
    for root, dirs, files in os.walk(dirpath):
        for file in files:
            if (file.endswith(('.wav', '.WAV')) and 'SA1' not in file and 'SA2'
                    not in file):
                all_filepaths.append(os.path.join(root, file))
    if not all_filepaths:
        raise ValueError(('no target file found, check your paths in '
                          'config.py'))
    state = random.getstate()
    random.seed(42)
    random.shuffle(all_filepaths)
    n_files = len(all_filepaths)
    i_min, i_max = round(n_files*lims[0]), round(n_files*lims[1])
    all_filepaths = all_filepaths[i_min:i_max]
    random.setstate(state)
    filepath = random.choice(all_filepaths)
    x, fs_old = sf.read(filepath)
    if fs_old != fs:
        x = resample(x, fs_old, fs, axis=0)
    return x, filepath


def load_brir(room_alias, angle):
    '''
    Load a BRIR given a room alias and a BRIR angle. Currently a BRIR from the
    SURREY database is loaded. See config.py to set the path to the SURREY
    database in your filesystem.

    Parameters:
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
    database_dir = config.SURREY_PATH
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
    room_dir = os.path.join(database_dir, room_folder, '16kHz')
    r = re.compile('CortexBRIR_.*s_%ideg_16k.wav' % angle)
    filenames = list(filter(r.match, os.listdir(room_dir)))
    if len(filenames) > 1:
        raise ValueError('more than one file matches pattern')
    elif len(filenames) == 0:
        raise ValueError(('could not find room %s with angle %i in '
                          'filesystem' % room_alias))
    filepath = os.path.join(room_dir, filenames[0])
    return sf.read(filepath)


def load_brirs(room_alias, angles=None):
    '''
    Load a BRIRs given a room alias and a list of angles. Currently a BRIR from
    the SURREY database is loaded. See config.py to set the path to the SURREY
    database in your filesystem.

    Parameters:
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
        database_dir = config.SURREY_PATH
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
        room_dir = os.path.join(database_dir, room_folder, '16kHz')
        for filename in os.listdir(room_dir):
            brir, fs = sf.read(os.path.join(room_dir, filename))
            brirs.append(brir)
            fss.append(fs)
    elif not angles:
        return [], None
    else:
        for angle in angles:
            brir, fs = load_brir(room_alias, angle)
            brirs.append(brir)
            fss.append(fs)
    if any(fs != fss[0] for fs in fss):
        raise ValueError('the brirs do not all have the same samplerate')
    return brirs, fss[0]


def load_random_noise(type_, n_samples, lims=None, fs=16e3):
    '''
    Load a random noise sample given a specific noise type. Currently a noise
    from the DCASE Challenge 2019 Task 1 development set it loaded. See
    config.py to set the path to the DCASE Challenge 2019 Task 1 development
    set in your filesystem.

    Parameters:
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

    Returns:
        x:
            Noise signal.
        filepath:
            Noise filepath in filesystem.
        indices:
            Starting and ending indices of the audio sample extracted from
            filepath.
    '''
    dirpath = config.DCASE_PATH
    all_filepaths = []
    m = re.match('^dcase_(.*)$', type_)
    if m is None:
        raise ValueError('type_ should start with dcase_')
    prefix = m.group(1)
    for root, dirs, files in os.walk(dirpath):
        for file in files:
            if file.endswith(('.wav', '.WAV')) and file.startswith(prefix):
                all_filepaths.append(os.path.join(root, file))
    if not all_filepaths:
        raise ValueError(('no noise file found, make sure the path to the '
                          'noise dataset is correctly set in config.py or '
                          'that type_ is either one of:\n'
                          '- dcase_airport\n'
                          '- dcase_bus\n'
                          '- dcase_metro\n'
                          '- dcase_park\n'
                          '- dcase_public_square\n'
                          '- dcase_shopping_mall\n'
                          '- dcase_street_pedestrian\n'
                          '- dcase_street_traffic\n'
                          '- dcase_tram'))
    if lims is not None:
        n_files = len(all_filepaths)
        i_min, i_max = round(lims[0]*n_files), round(lims[1]*n_files)
        all_filepaths = all_filepaths[i_min:i_max]
    filepath = random.choice(all_filepaths)
    x, fs_old = sf.read(filepath)
    if x.ndim == 2:
        x = x[:, 0]
    if fs_old != fs:
        x = resample(x, fs_old, fs)
    i_start = random.randint(0, len(x) - n_samples)
    i_end = i_start+n_samples
    x = x[i_start:i_end]
    return x, filepath, (i_start, i_end)
