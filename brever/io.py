import os
import re
import random
import sys

import numpy as np
import soundfile as sf
import sofa
import scipy.signal

from .config import defaults
from .utils import resample


def get_path(field_name, def_cfg=None):
    if def_cfg is None:
        def_cfg = defaults()
    output = getattr(def_cfg.PATH, field_name)
    if not os.path.exists(output):
        raise ValueError('the following dataset path was not found in the '
                         f'filesystem: {output}')
    return output


def load_random_target(speaker, lims=None, fs=16e3, randomizer=None,
                       def_cfg=None, all_filepaths=None):
    '''
    Load a random target signal given a set of speaker aliases to draw from.

    Parameters:
        speaker:
            Speaker alias. Case insensitive. Can also be a regular expression. 
            Examples:
            - timit_FCJF0
            - timit_fcjf0
            - libri_19
            - timit_f.*
            - ieee
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
        def_cfg:
            Default configuration `~.config.AttrDict` object usually returned
            by `~.config.defaults`. Default is `None` which means
            `~.config.defaults` will be called.

    Returns:
        x:
            Audio signal.
        filepath:
            Path to the loaded file.
    '''
    if all_filepaths is None:
        all_filepaths = get_all_filepaths(speaker, def_cfg)
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


def load_brirs(room_alias, angles=None, fs=16e3, def_cfg=None):
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
            - 'ash_rXX' with XX ranging from 01 to 39
        angles:
            Angles from which the BRIRs were recorded.
        fs:
            Sampling frequency to resample the BRIR to.
        def_cfg:
            Default configuration `~.config.AttrDict` object usually returned
            by `~.config.defaults`. Default is `None` which means
            `~.config.defaults` will be called.

    Returns:
        brirs:
            List of BRIRs.
        fs:
            BRIRs original sampling frequency.
    '''
    if angles is None:
        angles = get_available_angles(room_alias, def_cfg)
        return load_brirs(room_alias, angles, fs, def_cfg)
    iterable = True
    try:
        iter(angles)
    except TypeError:
        iterable = False
    if iterable:
        brirs = []
        fss = []
        for angle in angles:
            brir, fs_ = load_brirs(room_alias, angle, fs, def_cfg)
            brirs.append(brir)
            fss.append(fs_)
        if not brirs:
            return [], None
        assert all(fs_ == fss[0] for fs_ in fss)
        return brirs, fss[0]
    angle = angles
    if room_alias.startswith('surrey_'):
        dirpath = get_path('SURREY', def_cfg)
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
        dirpath = get_path('HUDDERSFIELD', def_cfg)
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
    elif room_alias.startswith('ash_'):
        dirpath = get_path('ASH', def_cfg)
        m = re.match('^ash_r(.*)$', room_alias)
        if m is None:
            raise ValueError(f'wrong room alias: {room_alias}')
        room_number = m.group(1)
        if room_number in ['05a', '05b', '05A', '05B']:
            room_number = room_number.upper()
            dirpath = os.path.join(dirpath, 'BRIRs', 'R05')
        else:
            dirpath = os.path.join(dirpath, 'BRIRs', f'R{room_number}')
        filename = f'BRIR_R{room_number}_P1_E0_A{angle}.wav'
        filepath = os.path.join(dirpath, filename)
        brir, fs_ = sf.read(filepath)
    else:
        raise ValueError(f'wrong room alias: {room_alias}')
    if fs_ != fs:
        brir = resample(brir, fs_, fs, axis=0)
    return brir, fs_


def load_random_noise(noise_alias, n_samples, lims=None, fs=16e3,
                      randomizer=None, def_cfg=None, ltas=None):
    '''
    Load a random noise recording.

    Parameters:
        noise_alias:
            Noise alias. Can be either:
            - 'dcase_airport'
            - 'dcase_bus'
            - 'dcase_metro'
            - 'dcase_metro_station'
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
        def_cfg:
            Default configuration `~.config.AttrDict` object usually returned
            by `~.config.defaults`. Default is `None` which means
            `~.config.defaults` will be called.

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
        dirpath = get_path('DCASE', def_cfg)
        all_filepaths = []
        m = re.match('^dcase_(.*)$', noise_alias)
        if m is None:
            raise ValueError(f'wrong noise type, got {noise_alias}')
        prefix = f'{m.group(1)}-'
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
    elif noise_alias.startswith('icra_'):
        dirpath = get_path('ICRA', def_cfg)
        all_filepaths = []
        m = re.match('^icra_(.*)$', noise_alias)
        if m is None:
            raise ValueError(f'wrong noise type, got {noise_alias}')
        i = m.group(1)
        filepath = os.path.join(dirpath, f'ICRA_{i}.wav')
        x, fs_old = sf.read(filepath)
        if x.ndim == 2:
            x = x[:, 0]
        if fs_old != fs:
            x = resample(x, fs_old, fs)
        if lims is not None:
            i_start = round(lims[0]*len(x))
            i_end = round(lims[1]*len(x))
            x = x[i_start:i_end]
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
    else:
        raise ValueError(f'wrong noise alias: {noise_alias}')
    return x, filepath, (i_start, i_end)


def get_available_angles(room_alias, def_cfg=None):
    if room_alias.startswith('surrey_'):
        dirpath = get_path('SURREY', def_cfg)
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
        r = re.compile(r'CortexBRIR_.*s_(-?\d{1,2})deg_16k\.wav')
        filenames = list(filter(r.match, os.listdir(room_dir)))
        angles = sorted(set(int(r.match(fn).group(1)) for fn in filenames))
    elif room_alias.startswith('huddersfield_'):
        dirpath = get_path('HUDDERSFIELD', def_cfg)
        m = re.match('^huddersfield_(.*)m$', room_alias)
        if m is None:
            raise ValueError(f'wrong room alias: {room_alias}')
        room_name = m.group(1)
        filename = f'{room_name.upper()}m.sofa'
        filepath = os.path.join(dirpath, 'Binaural', 'SOFA', filename)
        HRTF = sofa.Database.open(filepath)
        positions = HRTF.Source.Position.get_values(system='spherical')
        angles = positions[:, 0]
    elif room_alias.startswith('ash_'):
        angles = []
        dirpath = get_path('ASH', def_cfg)
        m = re.match('^ash_r(.*)$', room_alias)
        if m is None:
            raise ValueError(f'wrong room alias: {room_alias}')
        room_number = m.group(1)
        if room_number in ['05a', '05b', '05A', '05B']:
            room_number = room_number.upper()
            dirpath = os.path.join(dirpath, 'BRIRs', 'R05')
        else:
            dirpath = os.path.join(dirpath, 'BRIRs', f'R{room_number}')
        for filename in os.listdir(dirpath):
            if filename.endswith('.wav'):
                m = re.match(f'BRIR_R{room_number}_P1_E0_A(.*).wav', filename)
                if m is None:
                    continue
                angles.append(int(m.group(1)))
        if not angles:
            raise ValueError(f'no brir found for room {room_alias}')
    else:
        raise ValueError(f'wrong room alias: {room_alias}')
    return angles


def get_rooms(regexps):
    avail_rooms = [
        'surrey_anechoic',
        'surrey_room_a',
        'surrey_room_b',
        'surrey_room_c',
        'surrey_room_d',
        'huddersfield_c1m',
        'huddersfield_c2m',
        'huddersfield_c4m',
        'huddersfield_c6m',
        'huddersfield_c8m',
        'huddersfield_l1m',
        'huddersfield_l2m',
        'huddersfield_l4m',
        'huddersfield_l6m',
        'huddersfield_l8m',
        'huddersfield_lw1m',
        'huddersfield_lw2m',
        'huddersfield_lw4m',
        'huddersfield_lw6m',
        'huddersfield_lw8m',
        'ash_r01',
        'ash_r02',
        'ash_r03',
        'ash_r04',
        'ash_r05a',
        'ash_r05b',
        'ash_r06',
        'ash_r07',
        'ash_r08',
        'ash_r09',
        'ash_r10',
        'ash_r11',
        'ash_r12',
        'ash_r13',
        'ash_r14',
        'ash_r15',
        'ash_r16',
        'ash_r17',
        'ash_r18',
        'ash_r19',
        'ash_r20',
        'ash_r21',
        'ash_r22',
        'ash_r23',
        'ash_r24',
        'ash_r25',
        'ash_r26',
        'ash_r27',
        'ash_r28',
        'ash_r29',
        'ash_r30',
        'ash_r31',
        'ash_r32',
        'ash_r33',
        'ash_r34',
        'ash_r35',
        'ash_r36',
        'ash_r37',
        'ash_r38',
        'ash_r39',
    ]
    output = set()
    for regexp in regexps:
        if not regexp.startswith('^'):
            regexp = f'^{regexp}'
        if not regexp.endswith('$'):
            regexp = f'{regexp}$'
        r = re.compile(regexp)
        rooms = list(filter(r.match, avail_rooms))
        if not rooms:
            raise ValueError(f'regular expression {regexp} does not match '
                             'with any room')
        for room in rooms:
            if room in output:
                raise ValueError('the list of supplied regular expressions '
                                 'for room aliases leads to sets of found '
                                 f'rooms that are not disjoint: {regexps}')
            output.add(room)
    return output


def get_all_filepaths(speaker, def_cfg=None):
    alias_to_key_map = {
        'timit': 'TIMIT',
        'libri': 'LIBRI',
        'ieee': 'IEEE',
    }
    dset_alias = speaker.split('_')[0]
    if dset_alias in alias_to_key_map.keys():
        dirpath = get_path(alias_to_key_map[dset_alias], def_cfg)
    else:
        raise ValueError(f'Wrong speaker alias: {dset_alias}')
    if not os.path.exists(dirpath):
        raise ValueError(f'Directory not found: {dirpath}')
    all_filepaths = []
    if dset_alias == 'timit':
        pattern = speaker.split('_')[1].lower()
        if not pattern.startswith('^'):
            pattern = f'^{pattern}'
        if not pattern.endswith('$'):
            pattern = f'{pattern}$'
        for root, dirs, files in os.walk(dirpath):
            basename = os.path.basename(root)
            if re.match(pattern, basename.lower()):
                for file in files:
                    if (file.lower().endswith('.wav')
                            and 'SA1' not in file
                            and 'SA2' not in file):
                        all_filepaths.append(os.path.join(root, file))
    elif dset_alias == 'ieee':
        for root, dirs, files in os.walk(dirpath):
            for file in files:
                if file.lower().endswith('.wav'):
                    all_filepaths.append(os.path.join(root, file)) 
    elif dset_alias == 'libri':
        pattern = speaker.split('_')[1].lower()
        if '^' not in pattern:
            pattern = f'^{pattern}'
        if '$' not in pattern:
            pattern = f'{pattern}$'
        for root, dirs, files in os.walk(dirpath):
            basename = os.path.basename(root)
            if re.match(pattern, basename.lower()):
                for dir_ in dirs:
                    subroot = os.path.join(root, dir_)
                    for file_ in os.listdir(subroot):
                        if file_.lower().endswith('.flac'):
                            all_filepaths.append(os.path.join(subroot, file_))
    if not all_filepaths:
        raise ValueError(f'No audio file found for speaker {speaker}')
    return all_filepaths


def get_average_duration(speaker, def_cfg=None):
    all_filepaths = get_all_filepaths(speaker, def_cfg)
    t = 0
    for filepath in all_filepaths:
        t += sf.info(filepath).duration
    return t/len(all_filepaths)


def get_ltas(speaker=None, all_filepaths=None, def_cfg=None, n_fft=512,
             n_overlap=256, n_oct=3, verbose=False):
    if speaker is None and all_filepaths is None:
        raise ValueError('must provided speaker or all filepaths')
    elif speaker is not None and all_filepaths is not None:
        raise ValueError('cannot provide both speaker and all filepaths')
    if all_filepaths is None:
        all_filepaths = get_all_filepaths(speaker, def_cfg)
    n = n_fft//2+1
    ltas = np.zeros(n)
    for i, filepath in enumerate(all_filepaths):
        if verbose:
            sys.stdout.write('\r')
            sys.stdout.write(f'Processing file {i+1}/{len(all_filepaths)}')
        x, _ = sf.read(filepath)
        _, _, X = scipy.signal.stft(x, nperseg=n_fft, noverlap=n_overlap)
        ltas += np.mean(np.abs(X)**2, axis=1)
    if verbose:
        sys.stdout.write('\n')
    f = np.arange(1, n)
    sigma = (f/n_oct)/np.pi
    df = np.subtract.outer(f, f)
    g = np.exp(-0.5*(df/sigma)**2)/(sigma*(2*np.pi)**0.5)
    g /= g.sum(axis=1)
    ltas_smooth = np.copy(ltas)
    ltas_smooth[1:] = g@ltas_smooth[1:]
    return ltas_smooth
