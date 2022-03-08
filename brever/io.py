import logging
import os
import random
import re

import numpy as np
import soundfile as sf
import sofa
import scipy.signal

from .config import get_config


def resample(x, old_fs, new_fs, axis=0):
    """
    Resampling.

    Resample an array given an original sample rate and a target sample rate
    along a given axis.

    Parameters
    ----------
    x : array_like
        Input array.
    old_fs : int
        Original samplerate.
    new_fs : int
        Target samplerate.
    axis : int
        Axis along which to resample.

    Returns
    -------
    y : array_like
        Resampled array.
    """
    ratio = new_fs/old_fs
    n_samples = int(np.ceil(x.shape[axis] * ratio))
    y = scipy.signal.resample(x, n_samples, axis=axis)
    return y


class AudioFileLoader:
    def __init__(self, fs=16e3, resample=True, lims_speech=[0.0, 1.0],
                 lims_noise=[0.0, 1.0]):
        self.load_cfg()
        self.fs = fs
        self.resample = resample
        self.lims_speech = lims_speech
        self.lims_noise = lims_noise
        self._speech_files = {}
        self._noise_files = {}
        self._room_angles = {}
        self._room_regexps = {}

    def load_cfg(self):
        self.path_cfg = get_config('config/paths.yaml')

    def get_path(self, alias):
        if self.path_cfg is None:
            self.load_cfg()
        try:
            output = getattr(self.path_cfg, alias.upper())
        except AttributeError:
            raise ValueError(f'wrong alias, got {alias}')
        if not os.path.exists(output):
            raise ValueError('the following dataset path was not found in the '
                             f'filesystem: {output}')
        return output

    def load_file(self, file):
        x, fs = sf.read(file)
        if x.ndim == 2:
            x = x[:, 0]
        if fs != self.fs:
            if not self.resample:
                raise ValueError(f'file {file} has wrong sampling rate, got '
                                 f'{fs}, expected {self.fs}')
            else:
                x = resample(x, fs, self.fs, axis=0)
        return x

    def load_noise(self, file, n_samples, use_lims=False):
        x = self.load_file(file)

        if use_lims:
            lims = self.lims_noise
            i_start = round(lims[0]*len(x))
            i_end = round(lims[1]*len(x))
            x = x[i_start:i_end]

        rand = random.Random(0)
        if len(x) < n_samples:
            i_start = rand.randint(0, n_samples)
            indices = np.arange(n_samples) + i_start
            indices = indices % len(x)
            x = x[indices]
            i_end = None
        else:
            i_start = rand.randint(0, len(x) - n_samples)
            i_end = i_start + n_samples
            x = x[i_start:i_end]

        return x, (i_start, i_end)

    def get_speech_files(self, speaker):
        if speaker in self._speech_files.keys():
            return self._speech_files[speaker]
        prefix = speaker.split('_')[0]
        dirpath = self.get_path(prefix)
        output = []
        if prefix in ['timit', 'libri']:
            try:
                regexp = speaker.split('_')[1].lower()
            except IndexError:
                raise ValueError(f'wrong speaker, got {speaker}')
            regexp = check_regexp(regexp)
            speakers = self.get_speakers(prefix)
            if prefix == 'timit':
                for key in filter(re.compile(regexp).match, speakers):
                    for file in os.listdir(speakers[key]):
                        if file.lower().endswith('.wav'):
                            output.append(os.path.join(speakers[key], file))
            elif prefix == 'libri':
                for key in filter(re.compile(regexp).match, speakers):
                    for root, dirs, files in os.walk(speakers[key]):
                        for file in files:
                            if file.lower().endswith('.flac'):
                                output.append(os.path.join(root, file))
        elif prefix in ['ieee', 'arctic', 'hint', 'vctk']:
            for root, dirs, files in os.walk(dirpath):
                for file in files:
                    if file.lower().endswith(('.wav', '.flac')):
                        output.append(os.path.join(root, file))
        else:
            raise ValueError(f'wrong speaker, got {speaker}')
        if not output:
            raise ValueError(f'no audio file found for speaker {speaker}')
        self._speech_files[speaker] = output
        return output

    def get_speakers(self, prefix):
        dirpath = self.get_path(prefix)
        if prefix == 'timit':
            male_speakers = []
            female_speakers = []
            for root, dirs, files in os.walk(dirpath):
                if root.endswith('TRAIN') or root.endswith('TEST'):
                    for dialect in dirs:
                        root_ = os.path.join(root, dialect)
                        for speaker_id in os.listdir(root_):
                            item = os.path.join(root_, speaker_id)
                            if speaker_id.startswith('M'):
                                male_speakers.append(item)
                            elif speaker_id.startswith('F'):
                                female_speakers.append(item)
                            else:
                                raise ValueError(f'Found unexpected TIMIT '
                                                 'speaker ID not starting '
                                                 f'with M nor F: {item}')
            male_speakers.sort()
            female_speakers.sort()
            output = {}
            for i, path in enumerate(male_speakers):
                output[f'm{i}'] = path
            for i, path in enumerate(female_speakers):
                output[f'f{i}'] = path
            return output
        elif prefix == 'libri':
            male_speakers = []
            female_speakers = []
            txt_file = os.path.join(dirpath, 'SPEAKERS.TXT')
            id_sex_dict = {}
            with open(txt_file) as f:
                for line in f.readlines():
                    if not line.startswith(';'):
                        id_ = line[:4].rstrip()
                        sex = line[7]
                        id_sex_dict[id_] = sex
            for dset_name in os.listdir(dirpath):
                dset_path = os.path.join(dirpath, dset_name)
                if os.path.isdir(dset_path):
                    for speaker_id in os.listdir(dset_path):
                        speaker_path = os.path.join(dset_path, speaker_id)
                        sex = id_sex_dict[speaker_id]
                        if sex == 'M':
                            male_speakers.append(speaker_path)
                        elif sex == 'F':
                            female_speakers.append(speaker_path)
                        else:
                            raise ValueError(f'Bad sex character pulled from '
                                             f'SPEAKERS.TXT: {sex}')
            male_speakers.sort(key=lambda x: int(os.path.basename(x)))
            female_speakers.sort(key=lambda x: int(os.path.basename(x)))
            output = {}
            for i, path in enumerate(male_speakers):
                output[f'm{i}'] = path
            for i, path in enumerate(female_speakers):
                output[f'f{i}'] = path
            return output

    def get_noise_files(self, noise):
        if noise in self._noise_files.keys():
            return self._noise_files[noise]
        prefix = noise.split('_')[0]
        dirpath = self.get_path(prefix)
        output = []
        if prefix == 'dcase':
            m = re.match(f'^{prefix}_(.*)$', noise)
            if m is None:
                raise ValueError(f'wrong noise type, got {noise}')
            regexp = f'{m.group(1)}'
            regexp = check_regexp(regexp)
            for root, dirs, files in os.walk(dirpath):
                for file in files:
                    if file.lower().endswith('.wav'):
                        noise_type = file.split('-')[0]
                        if re.match(regexp, noise_type):
                            output.append(os.path.join(root, file))
        elif prefix == 'icra':
            m = re.match(f'^{prefix}_(.*)$', noise)
            if m is None:
                raise ValueError(f'wrong noise type, got {noise}')
            regexp = f'{m.group(1)}'
            regexp = check_regexp(regexp)
            for root, dirs, files in os.walk(dirpath):
                for file in files:
                    if file.lower().endswith('.wav'):
                        m = re.match('^ICRA_(.*).wav$', file)
                        if m is not None:
                            icra_number = m.group(1)
                            if re.match(regexp, icra_number):
                                output.append(os.path.join(root, file))
        elif prefix == 'arte':
            to_find = [
                '01_Library_binaural_withEQ.wav',
                '02_Office_binaural_withEQ.wav',
                '03_Church_1_binaural_withEQ.wav',
                '04_Living_Room_binaural_withEQ.wav',
                '05_Church_2_binaural_withEQ.wav',
                '06_Diffuse_noise_binaural_withEQ.wav',
                '07_Cafe_1_binaural_withEQ.wav',
                '08_Cafe_2_binaural_withEQ.wav',
                '09_Dinner_party_binaural_withEQ.wav',
                '10_Street_Balcony_binaural_withEQ.wav',
                '11_Train_Station_binaural_withEQ.wav',
                '12_Food_Court_1_binaural_withEQ.wav',
                '13_Food_Court_2_binaural_withEQ.wav',
            ]
            for file in to_find:
                found = False
                for root, dirs, files in os.walk(dirpath):
                    if file in files:
                        output.append(os.path.join(root, file))
                        found = True
                        break
                if not found:
                    raise ValueError('the ARTE database in the filesystem '
                                     f'is incomplete, could not find {file}')
        elif prefix == 'demand':
            for root, dirs, files in os.walk(dirpath):
                for file in files:
                    if file.endswith('ch01.wav'):
                        output.append(os.path.join(dirpath, file))
        elif prefix == 'noisex':
            m = re.match('^noisex_(.*)$', noise)
            if m is None:
                raise ValueError(f'wrong noise type, got {noise}')
            regexp = f'{m.group(1)}'
            regexp = check_regexp(regexp)
            for file in os.listdir(dirpath):
                if file.endswith('.wav') and re.match(regexp, file[:-4]):
                    output.append(os.path.join(dirpath, file))
        else:
            raise ValueError(f'wrong noise alias, got {noise}')
        if not output:
            raise ValueError(f'no audio file found for noise {noise}')
        self._noise_files[noise] = output
        return output

    def load_brirs(self, room, angles=None):
        if angles is None:
            angles = self.get_angles(room)
            return self.load_brirs(room, angles)

        if isinstance(angles, list):
            if not angles:
                raise ValueError('angles cannot be an empty list')
            brirs = []
            files = []
            for angle in angles:
                brir, file = self.load_brirs(room, angle)
                brirs.append(brir)
                files.append(file)
            return brirs, files

        if not isinstance(angles, (int, float)):
            raise TypeError('angles must be None, list, float or int, got '
                            f'{type(angles).__name__}')

        angle = angles
        prefix = room.split('_')[0]
        dirpath = self.get_path(prefix)
        m = re.match(f'^{prefix}_(.*)$', room)
        if m is None:
            raise ValueError(f'wrong room alias, got {room}')

        if prefix == 'surrey':
            if m.group(1) == 'anechoic':
                room_folder = 'Anechoic'
            else:
                m = re.match('^room_(.)$', m.group(1))
                if m is None:
                    raise ValueError(f'wrong room alias, got {room}')
                room_letter = m.group(1)
                room_folder = 'Room_%s' % room_letter.upper()
            room_dir = os.path.join(dirpath, room_folder, '16kHz')
            r = re.compile('CortexBRIR_.*s_%ideg_16k.wav' % angle)
            files = list(filter(r.match, os.listdir(room_dir)))
            if len(files) > 1:
                raise ValueError('more than one brir was found for room '
                                 f'{room} and angle {angle} in filesystem')
            elif len(files) == 0:
                raise ValueError('could not find any brir for room '
                                 f'{room} and angle {angle} in filesystem')
            file = os.path.join(room_dir, files[0])
            brir, fs = sf.read(file)
        elif prefix == 'huddersfield':
            file = f'{m.group(1).upper()}m.sofa'
            file = os.path.join(dirpath, 'Binaural', 'SOFA', file)
            HRTF = sofa.Database.open(file)
            positions = HRTF.Source.Position.get_values(system='spherical')
            measurement = np.argwhere(abs(positions[:, 0] - angle) <= 1e-6)
            if len(measurement) == 0:
                raise ValueError('could not find any brir for room '
                                 f'{room} and angle {angle} in filesystem')
            elif len(measurement) > 1:
                raise ValueError('more than one brir was found for room '
                                 f'{room} and angle {angle} in filesystem')
            measurement = int(measurement)
            ir_l = HRTF.Data.IR.get_values({'M': measurement, 'R': 0})
            ir_r = HRTF.Data.IR.get_values({'M': measurement, 'R': 1})
            brir = np.vstack((ir_l, ir_r)).T
            fs = HRTF.Data.SamplingRate.get_values(indices={'M': measurement})
        elif prefix == 'ash':
            m = re.match('^ash_r(.*)$', room)
            if m is None:
                raise ValueError(f'wrong room alias: {room}')
            room_number = m.group(1)
            if room_number in ['05a', '05b', '05A', '05B']:
                room_number = room_number.upper()
                dirpath = os.path.join(dirpath, 'BRIRs', 'R05')
            else:
                dirpath = os.path.join(dirpath, 'BRIRs', f'R{room_number}')
            file = f'BRIR_R{room_number}_P1_E0_A{angle}.wav'
            file = os.path.join(dirpath, file)
            brir, fs = sf.read(file)
        elif prefix == 'air':
            # inconsistency in angle directions in AACHEN dataset!
            # for aula_carolina, angles go from left (0) to right (180)
            # for stairway, angles go from right (0) to left (180)
            if m.group(1).startswith('aula_carolina'):
                angle = angle + 90
                file = f'air_binaural_{m.group(1)}_{angle}_3.wav'
            elif m.group(1).startswith('stairway'):
                angle = - angle + 90
                file = f'air_binaural_{m.group(1)}_{angle}.wav'
            else:
                file = f'air_binaural_{m.group(1)}.wav'
            brir, fs = sf.read(os.path.join(dirpath, file))
        elif prefix == 'catt':
            m = re.match('^catt_([0-9])([0-9])$', room)
            i, j = m.group(1), m.group(2)
            file = f'CATT_{i}_{j}s_{angle}.wav'
            file = os.path.join(dirpath, f'{i}_{j}s', file)
            brir, fs = sf.read(file)
        elif prefix == 'avil':
            angle = (360 - angle) % 360
            file = f'{m.group(1)}_azim_{angle}_degree.wav'
            file = os.path.join(dirpath, m.group(1), file)
            brir, fs = sf.read(file)
        elif prefix == 'elospheres':
            room_name = m.group(1)[0].upper() + m.group(1)[1:]
            file = os.path.join(dirpath, f'{room_name}.sofa')
            HRTF = sofa.Database.open(file)
            if room_name == 'Car':
                angles = [(-90 - 2.5*i) for i in range(36)] + \
                         [(180 - 2.5*i) for i in range(37)]
            else:
                angles = [90 - 2.5*i for i in range(73)]
            measurement = angles.index(angle)
            ir_l = HRTF.Data.IR.get_values({'M': measurement, 'R': 0, 'E': 1})
            ir_r = HRTF.Data.IR.get_values({'M': measurement, 'R': 1, 'E': 1})
            brir = np.vstack((ir_l, ir_r)).T
            fs = HRTF.Data.SamplingRate.get_values(indices={'M': measurement})
        elif prefix == 'bras':
            scene_name = m.group(1).upper()
            to_find = f'{scene_name}_BRIRs.sofa'
            found = []
            for root, folder, files in os.walk(dirpath):
                if to_find in files:
                    found.append(os.path.join(root, to_find))
            if not found:
                raise ValueError(f'could not find {room} BRIRs in filesystem')
            elif len(found) > 1:
                raise ValueError(f'found more than one match for alias {room}')
            file, = found
            HRTF = sofa.Database.open(file)
            angles = [-44 + 2*i for i in range(45)]
            measurement = angles.index(angle)
            if scene_name in ['CR2', 'CR3', 'CR4']:
                E = 4
            else:
                E = 0
            ir_l = HRTF.Data.IR.get_values({'M': measurement, 'R': 0, 'E': E})
            ir_r = HRTF.Data.IR.get_values({'M': measurement, 'R': 1, 'E': E})
            brir = np.vstack((ir_l, ir_r)).T
            fs = HRTF.Data.SamplingRate.get_values(indices={'M': measurement})
        else:
            raise ValueError(f'wrong room alias, got {room}')
        if fs != self.fs:
            if not self.resample:
                raise ValueError(f'file {file} has wrong sampling rate, got '
                                 f'{fs}, expected {self.fs}')
            else:
                brir = resample(brir, fs, self.fs, axis=0)
        return brir, file

    def get_angles(self, room):
        if room in self._room_angles.keys():
            return self._room_angles[room]
        prefix = room.split('_')[0]
        dirpath = self.get_path(prefix)
        m = re.match(f'^{prefix}_(.*)$', room)
        if m is None:
            raise ValueError(f'wrong room alias, got {room}')

        if prefix == 'surrey':
            if m.group(1) == 'anechoic':
                room_folder = 'Anechoic'
            else:
                m = re.match('^room_(.)$', m.group(1))
                if m is None:
                    raise ValueError(f'wrong room alias: {room}')
                room_letter = m.group(1)
                room_folder = 'Room_%s' % room_letter.upper()
            room_dir = os.path.join(dirpath, room_folder, '16kHz')
            r = re.compile(r'CortexBRIR_.*s_(-?\d{1,2})deg_16k\.wav')
            files = list(filter(r.match, os.listdir(room_dir)))
            angles = [int(r.match(f).group(1)) for f in files]
        elif prefix == 'huddersfield':
            file = f'{m.group(1).upper()}m.sofa'
            file = os.path.join(dirpath, 'Binaural', 'SOFA', file)
            HRTF = sofa.Database.open(file)
            positions = HRTF.Source.Position.get_values(system='spherical')
            angles = positions[:, 0]
        elif prefix == 'ash':
            angles = []
            m = re.match('^ash_r(.*)$', room)
            if m is None:
                raise ValueError(f'wrong room alias: {room}')
            room_num = m.group(1)
            if room_num in ['05a', '05b', '05A', '05B']:
                room_num = room_num.upper()
                dirpath = os.path.join(dirpath, 'BRIRs', 'R05')
            else:
                dirpath = os.path.join(dirpath, 'BRIRs', f'R{room_num}')
            for file in os.listdir(dirpath):
                if file.endswith('.wav'):
                    m = re.match(f'BRIR_R{room_num}_P1_E0_A(.*).wav', file)
                    if m is None:
                        continue
                    angles.append(int(m.group(1)))
            if not angles:
                raise ValueError(f'no brir found for room {room}')
        elif prefix == 'air':
            if m.group(1) == 'aula_carolina_1_3':
                angles = [-90, -45, 0, 45, 90]
            elif m.group(1).startswith('stairway'):
                angles = [-90, -75, -60, -45, -30, -15,
                          0, 15, 30, 45, 60, 75, 90]
            else:
                angles = [0]
        elif prefix == 'catt':
            m = re.match('^catt_([0-9])([0-9])$', room)
            if m is None:
                raise ValueError(f'wrong room alias: {room}')
            i, j = m.group(1), m.group(2)
            folder = os.path.join(dirpath, f'{i}_{j}s')
            r = re.compile(rf'^CATT_{i}_{j}s_(-?\d{{1,2}}).wav$')
            angles = [int(r.match(f).group(1)) for f in os.listdir(folder)]
        elif prefix == 'avil':
            folder = os.path.join(dirpath, m.group(1))
            r = re.compile(rf'^{m.group(1)}_azim_(\d{{1,3}})_degree.wav$')
            angles = [int(r.match(f).group(1)) for f in os.listdir(folder)]
            angles = [-((a + 180) % 360) + 180 for a in angles]
        elif prefix == 'elospheres':
            if m.group(1) == 'car':
                angles = [(-90 - 2.5*i) for i in range(36)] + \
                         [(180 - 2.5*i) for i in range(37)]
            else:
                angles = [90 - 2.5*i for i in range(73)]
        elif prefix == 'bras':
            angles = [-44 + 2*i for i in range(45)]
        else:
            raise ValueError(f'wrong room alias: {room}')

        self._room_angles[room] = angles
        return angles

    def get_duration(self, speaker):
        files = self.get_speech_files(speaker)
        duration = 0
        for file in files:
            duration += sf.info(file).duration
        return duration, len(files)

    def calc_weights(self, speakers):
        weights = {}
        if len(speakers) > 1:
            logging.info('Calculating corpus probabilities')
            for speaker in speakers:
                duration, n_files = self.get_duration(speaker)
                weights[speaker] = n_files/duration
            logging.info(weights)
        else:
            speaker, = speakers
            weights[speaker] = 1
        return weights

    def calc_ltas(self, speakers=None, n_fft=512, n_overlap=256, n_oct=3):
        if isinstance(speakers, (list, set)):
            if not speakers:
                raise ValueError('speakers cannot be an empty list or set')
            files = []
            for speaker in speakers:
                files += self.get_speech_files(speaker)
        elif isinstance(speakers, str):
            files = self.get_speech_files(speakers)
        else:
            raise TypeError('speakers must be str, list or set, got '
                            f'{type(speakers).__name__}')
        logging.info(f'Calculating LTAS from {len(files)} files')
        n = n_fft//2+1
        ltas = np.zeros(n)
        for i, file in enumerate(files):
            x, _ = sf.read(file)
            _, _, X = scipy.signal.stft(x, nperseg=n_fft, noverlap=n_overlap)
            ltas += np.mean(np.abs(X)**2, axis=1)
        f = np.arange(1, n)
        sigma = (f/n_oct)/np.pi
        df = np.subtract.outer(f, f)
        g = np.exp(-0.5*(df/sigma)**2)/(sigma*(2*np.pi)**0.5)
        g /= g.sum(axis=1)
        ltas_smooth = np.copy(ltas)
        ltas_smooth[1:] = g@ltas_smooth[1:]
        return ltas_smooth

    def get_rooms(self, regexp):
        # TODO: implement distance option to reduce huddersfield and air rooms
        # TODO: check huddersfield support
        avail_rooms = [
            'surrey_anechoic',
            'surrey_room_a',
            'surrey_room_b',
            'surrey_room_c',
            'surrey_room_d',
            # 'huddersfield_c1m',
            # 'huddersfield_c2m',
            # 'huddersfield_c4m',
            # 'huddersfield_c6m',
            # 'huddersfield_c8m',
            # 'huddersfield_l1m',
            # 'huddersfield_l2m',
            # 'huddersfield_l4m',
            # 'huddersfield_l6m',
            # 'huddersfield_l8m',
            # 'huddersfield_lw1m',
            # 'huddersfield_lw2m',
            # 'huddersfield_lw4m',
            # 'huddersfield_lw6m',
            # 'huddersfield_lw8m',
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
            # 'ash_r17',  # few angles between -90 and 90
            'ash_r18',
            'ash_r19',
            # 'ash_r20',  # only 5 angles available
            'ash_r21',
            # 'ash_r22',  # few angles between -90 and 90
            'ash_r23',
            'ash_r24',
            'ash_r25',
            'ash_r26',
            # 'ash_r27',  # only 5 angles available
            'ash_r28',
            'ash_r29',
            'ash_r30',
            'ash_r31',
            # 'ash_r32',  # few angles between -90 and 90
            'ash_r33',
            'ash_r34',
            'ash_r35',
            'ash_r36',
            'ash_r37',
            'ash_r38',
            'ash_r39',
            'air_aula_carolina_1_1',
            'air_aula_carolina_1_2',
            'air_aula_carolina_1_3',
            'air_aula_carolina_1_4',
            'air_aula_carolina_1_5',
            'air_aula_carolina_1_6',
            'air_aula_carolina_1_7',
            'air_booth_0_1',
            'air_booth_0_2',
            'air_booth_0_3',
            'air_booth_1_1',
            'air_booth_1_2',
            'air_booth_1_3',
            'air_lecture_0_1',
            'air_lecture_0_2',
            'air_lecture_0_3',
            'air_lecture_0_4',
            'air_lecture_0_5',
            'air_lecture_0_6',
            'air_lecture_1_1',
            'air_lecture_1_2',
            'air_lecture_1_3',
            'air_lecture_1_4',
            'air_lecture_1_5',
            'air_lecture_1_6',
            'air_meeting_0_1',
            'air_meeting_0_2',
            'air_meeting_0_3',
            'air_meeting_0_4',
            'air_meeting_0_5',
            'air_meeting_1_1',
            'air_meeting_1_2',
            'air_meeting_1_3',
            'air_meeting_1_4',
            'air_meeting_1_5',
            'air_office_0_1',
            'air_office_0_2',
            'air_office_0_3',
            'air_office_1_1',
            'air_office_1_2',
            'air_office_1_3',
            'air_stairway_1_1',
            'air_stairway_1_2',
            'air_stairway_1_3',
            'catt_00',
            'catt_01',
            'catt_02',
            'catt_03',
            'catt_04',
            'catt_05',
            'catt_06',
            'catt_07',
            'catt_08',
            'catt_09',
            'catt_10',
            'avil_anechoic',
            'avil_high',
            'avil_low',
            'avil_medium',
            'elospheres_anechoic',
            'elospheres_restaurant',
            'elospheres_kitchen',
            # 'elospheres_car',  # car should not be used because not centered
            # 'bras_cr1',  # cr1 should not be used because not centered
            'bras_cr2',
            'bras_cr3',
            'bras_cr4',
            # 'bras_rs1_absorbing',
            # 'bras_rs1_diffuse',
            # 'bras_rs1_rigid',
            # 'bras_rs3',  # rs3 should not be used because not centered
            'bras_rs5',
        ]
        if regexp in self._room_regexps.keys():
            return self._room_regexps[regexp]
        r = re.compile(regexp)
        rooms = list(filter(r.match, avail_rooms))
        if not rooms:
            raise ValueError(f'regular expression {regexp} does not match '
                             'with any room')
        rooms = set(rooms)
        self._room_regexps[regexp] = rooms
        return set(rooms)

    def scan_material(self, speakers, noises, room_regexps):
        # save all the paths now
        for regexp in room_regexps:
            rooms = self.get_rooms(regexp)
            for room in rooms:
                self.get_angles(room)
        for speaker in speakers:
            self.get_speech_files(speaker)
        for noise in noises:
            self.get_noise_files(noise)


def check_regexp(regexp):
    if not regexp.startswith('^'):
        regexp = f'^{regexp}'
    if not regexp.endswith('$'):
        regexp = f'{regexp}$'
    return regexp
