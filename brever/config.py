import sys
import yaml


class Struct:
    def __init__(self, data=None):
        if data:
            for key, value in data.items():
                setattr(self, key, self._wrap(value))

    def _wrap(self, value):
        if isinstance(value, (list, tuple)):
            return type(value)([self._wrap(item) for item in value])
        else:
            if isinstance(value, dict):
                return Struct(value)
            else:
                return value

    def update(self, data):
        for key, value in data.items():
            if key not in self.__dict__:
                raise AttributeError((f'{type(self).__name__} instance has no '
                                      f'attribute {key}'))
            elif isinstance(getattr(self, key), Struct):
                if not isinstance(value, dict):
                    raise TypeError((f'value with key {key} must have type '
                                     f'dict'))
                getattr(self, key).update(value)
            else:
                if type(getattr(self, key)) != type(value):
                    raise TypeError((f'value with key {key} must have type '
                                     f'{type(getattr(self, key)).__name__}'))
                setattr(self, key, value)


PATH = Struct()
MIXTURES = Struct()
MIXTURES.RANDOM = Struct()
MIXTURES.RANDOM.RMSDB = Struct()
MIXTURES.RANDOM.SOURCES = Struct()
MIXTURES.RANDOM.TARGET = Struct()
MIXTURES.RANDOM.TARGET.ANGLE = Struct()
MIXTURES.RANDOM.TARGET.SNR = Struct()
MIXTURES.RANDOM.SOURCES.NUMBER = Struct()
MIXTURES.RANDOM.SOURCES.ANGLE = Struct()
MIXTURES.RANDOM.SOURCES.SNR = Struct()
MIXTURES.RANDOM.DIFFUSE = Struct()
MIXTURES.FILELIMITS = Struct()
FILTERBANK = Struct()
FRAMER = Struct()


PATH.DCASE = 'data\\audio\\TAU-urban-acoustic-scenes-2019-development'
PATH.SURREY = 'data\\brirs\\SURREY'
PATH.TIMIT = 'data\\audio\\TIMIT\\TRAIN'

FS = 16000

MIXTURES.NUMBER = 100
MIXTURES.PADDING = 0.1
MIXTURES.REFLECTIONBOUNDARY = 50e-3
MIXTURES.SCALERMS = True
MIXTURES.SAVE = True
MIXTURES.FILELIMITS.TARGET = (0.0, 1.0)
MIXTURES.FILELIMITS.NOISE = (0.0, 1.0)
MIXTURES.RANDOM.ROOMS = [
    'surrey_room_a',
    'surrey_room_b',
    'surrey_room_c',
    'surrey_room_d',
]
MIXTURES.RANDOM.RMSDB.MIN = -30
MIXTURES.RANDOM.RMSDB.MAX = -10
MIXTURES.RANDOM.TARGET.ANGLE.MIN = -90
MIXTURES.RANDOM.TARGET.ANGLE.MAX = 90
MIXTURES.RANDOM.TARGET.ANGLE.STEP = 5
MIXTURES.RANDOM.TARGET.SNR.MIN = 0
MIXTURES.RANDOM.TARGET.SNR.MAX = 15
MIXTURES.RANDOM.SOURCES.NUMBER.MIN = 0
MIXTURES.RANDOM.SOURCES.NUMBER.MAX = 3
MIXTURES.RANDOM.SOURCES.ANGLE.MIN = -90
MIXTURES.RANDOM.SOURCES.ANGLE.MAX = 90
MIXTURES.RANDOM.SOURCES.ANGLE.STEP = 5
MIXTURES.RANDOM.SOURCES.SNR.MIN = -5
MIXTURES.RANDOM.SOURCES.SNR.MAX = 5
MIXTURES.RANDOM.SOURCES.TYPES = [
    'dcase_airport',
    'dcase_bus',
    'dcase_metro',
    'dcase_park',
    'dcase_public_square',
    'dcase_shopping_mall',
    'dcase_street_pedestrian',
    'dcase_street_traffic',
    'dcase_tram',
]
MIXTURES.RANDOM.DIFFUSE.TYPES = [
    'noise_pink'
]

FILTERBANK.KIND = 'mel'
FILTERBANK.NFILTERS = 64
FILTERBANK.FMIN = 50
FILTERBANK.FMAX = 8000
FILTERBANK.FS = 16000
FILTERBANK.ORDER = 4

FRAMER.FRAMELENGTH = 512
FRAMER.HOPLENGTH = 256
FRAMER.WINDOW = 'hann'
FRAMER.CENTER = False

FEATURES = [
    'ild',
    'itd_ic',
    'mfcc',
    'pdf',
]

LABEL = 'irm'


def update(filename):
    with open(filename, 'r') as f:
        entries = yaml.safe_load(f)
    config = sys.modules[__name__]
    for key, value in entries.items():
        if isinstance(value, dict):
            getattr(config, key).update(value)
        else:
            setattr(config, key, value)
