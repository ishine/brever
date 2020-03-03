import os
import re
import soundfile as sf
import random
from resampy import resample

from . import config


def load_random_target(fs=16e3):
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
        room_dir = os.path.join(database_dir, room_folder, '16kHz')
        r = re.compile('CortexBRIR_.*s_%ideg_16k.wav' % angle)
        filenames = list(filter(r.match, os.listdir(room_dir)))
        if len(filenames) > 1:
            raise ValueError('more than one file matches pattern')
        filepath = os.path.join(room_dir, filenames[0])
        return sf.read(filepath)


def load_brirs(room_alias):
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
        room_dir = os.path.join(database_dir, room_folder, '16kHz')
        brirs = []
        fss = []
        for filename in os.listdir(room_dir):
            brir, fs = sf.read(os.path.join(room_dir, filename))
            brirs.append(brir)
            fss.append(fs)
        if any(fs != fss[0] for fs in fss):
            raise ValueError('the brirs do not all have the same samplerate')
        return brirs, fss[0]
