import os
import re
import soundfile as sf
import random
from resampy import resample

from . import config


def load_random_target(fs=16e3):
    '''
    Load a random target signal. Currently a random sentence from the EMIME
    databases is loaded. See config.py to set the path to the EMIME database
    in your filesystem.

    Parameters:
        fs:
            Sampling rating to resample the signal to. If it matches the
            original sampling rate, the signal is returned as is.

    Returns:
        x:
            Audio signal.
        filepath:
            Path to the loaded file.
    '''
    dirpath = config.EMIME_PATH
    all_filepaths = []
    for root, dirs, files in os.walk(dirpath):
        for file in files:
            all_filepaths.append(os.path.join(root, file))
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
    if m is not None:
        database_dir = config.SURREY_PATH
        room_name = m.group(1)
        m = re.match('^room_(.)$', room_name)
        if m is not None:
            room_letter = m.group(1)
            room_folder = 'Room_%s' % room_letter.upper()
        elif room_name == 'anechoic':
            room_folder = 'Anechoic'
        else:
            raise ValueError(('could not find room %s in filesystem'
                              % room_alias))
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
    else:
        raise ValueError('could not find room %s in filesystem' % room_alias)


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
        if m is not None:
            database_dir = config.SURREY_PATH
            room_name = m.group(1)
            m = re.match('^room_(.)$', room_name)
            if m is not None:
                room_letter = m.group(1)
                room_folder = 'Room_%s' % room_letter.upper()
            elif room_name == 'anechoic':
                room_folder = 'Anechoic'
            else:
                raise ValueError(('could not find room %s in filesystem'
                                  % room_alias))
            room_dir = os.path.join(database_dir, room_folder, '16kHz')
            for filename in os.listdir(room_dir):
                brir, fs = sf.read(os.path.join(room_dir, filename))
                brirs.append(brir)
                fss.append(fs)
        else:
            raise ValueError(('could not find room %s in filesystem'
                              % room_alias))
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
