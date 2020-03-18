class Struct:
    def __init__(self):
        pass

    def update(self, data):
        if not data:
            return
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

    def todict(self):
        data = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Struct):
                data[key] = value.todict()
            else:
                data[key] = value
        return data


def defaults():
    config = Struct()
    config.PRE = Struct()
    config.PRE.MIXTURES = Struct()
    config.PRE.MIXTURES.PATH = Struct()
    config.PRE.MIXTURES.FILELIMITS = Struct()
    config.PRE.MIXTURES.RANDOM = Struct()
    config.PRE.MIXTURES.RANDOM.RMSDB = Struct()
    config.PRE.MIXTURES.RANDOM.TARGET = Struct()
    config.PRE.MIXTURES.RANDOM.TARGET.ANGLE = Struct()
    config.PRE.MIXTURES.RANDOM.TARGET.SNR = Struct()
    config.PRE.MIXTURES.RANDOM.SOURCES = Struct()
    config.PRE.MIXTURES.RANDOM.SOURCES.NUMBER = Struct()
    config.PRE.MIXTURES.RANDOM.SOURCES.ANGLE = Struct()
    config.PRE.MIXTURES.RANDOM.SOURCES.SNR = Struct()
    config.PRE.MIXTURES.RANDOM.DIFFUSE = Struct()
    config.PRE.FILTERBANK = Struct()
    config.PRE.FRAMER = Struct()
    config.PRE.PLOT = Struct()
    config.POST = Struct()
    config.POST.PATH = Struct()
    config.MODEL = Struct()
    config.MODEL.TRAIN = Struct()
    config.MODEL.TRAIN.EARLYSTOP = Struct()

    config.PRE.FS = 16000
    config.PRE.MIXTURES.PATH.DCASE = 'data\\external\\DCASE'
    config.PRE.MIXTURES.PATH.SURREY = 'data\\external\\SURREY'
    config.PRE.MIXTURES.PATH.TIMIT = 'data\\external\\TIMIT\\TRAIN'
    config.PRE.MIXTURES.NUMBER = 100
    config.PRE.MIXTURES.PADDING = 0.1
    config.PRE.MIXTURES.REFLECTIONBOUNDARY = 50e-3
    config.PRE.MIXTURES.SAVE = False
    config.PRE.MIXTURES.FILELIMITS.TARGET = [0.0, 1.0]
    config.PRE.MIXTURES.FILELIMITS.NOISE = [0.0, 1.0]
    config.PRE.MIXTURES.RANDOM.ROOMS = [
        'surrey_room_a',
        'surrey_room_b',
        'surrey_room_c',
        'surrey_room_d',
    ]
    config.PRE.MIXTURES.RANDOM.RMSDB.MIN = -20
    config.PRE.MIXTURES.RANDOM.RMSDB.MAX = 0
    config.PRE.MIXTURES.RANDOM.TARGET.ANGLE.MIN = -90
    config.PRE.MIXTURES.RANDOM.TARGET.ANGLE.MAX = 90
    config.PRE.MIXTURES.RANDOM.TARGET.ANGLE.STEP = 5
    config.PRE.MIXTURES.RANDOM.TARGET.SNR.MIN = 0
    config.PRE.MIXTURES.RANDOM.TARGET.SNR.MAX = 15
    config.PRE.MIXTURES.RANDOM.SOURCES.NUMBER.MIN = 0
    config.PRE.MIXTURES.RANDOM.SOURCES.NUMBER.MAX = 3
    config.PRE.MIXTURES.RANDOM.SOURCES.ANGLE.MIN = -90
    config.PRE.MIXTURES.RANDOM.SOURCES.ANGLE.MAX = 90
    config.PRE.MIXTURES.RANDOM.SOURCES.ANGLE.STEP = 5
    config.PRE.MIXTURES.RANDOM.SOURCES.SNR.MIN = -5
    config.PRE.MIXTURES.RANDOM.SOURCES.SNR.MAX = 5
    config.PRE.MIXTURES.RANDOM.SOURCES.TYPES = [
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
    config.PRE.MIXTURES.RANDOM.DIFFUSE.TYPES = [
        'noise_pink'
    ]
    config.PRE.SCALERMS = True
    config.PRE.FILTERBANK.KIND = 'mel'
    config.PRE.FILTERBANK.NFILTERS = 64
    config.PRE.FILTERBANK.FMIN = 50
    config.PRE.FILTERBANK.FMAX = 8000
    config.PRE.FILTERBANK.ORDER = 4
    config.PRE.FRAMER.FRAMELENGTH = 512
    config.PRE.FRAMER.HOPLENGTH = 256
    config.PRE.FRAMER.WINDOW = 'hann'
    config.PRE.FRAMER.CENTER = False
    config.PRE.FEATURES = [
        'ild',
        'itd_ic',
        'mfcc',
        'pdf',
    ]
    config.PRE.LABEL = 'irm'
    config.PRE.PLOT.ON = True
    config.PRE.PLOT.NSAMPLES = 1000

    config.POST.PATH.TRAIN = 'data\\processed\\training'
    config.POST.PATH.VAL = 'data\\processed\\validation'
    config.POST.LOAD = False
    config.POST.FEATURES = [
        'ild',
        # 'itd_ic',
        'mfcc',
        # 'pdf',
    ]
    config.POST.STACK = 4
    config.POST.GLOBALSTANDARDIZATION = True
    config.POST.DECIMATION = 2

    config.MODEL.TRAIN.CUDA = True
    config.MODEL.TRAIN.BATCHSIZE = 32
    config.MODEL.TRAIN.SHUFFLE = True
    config.MODEL.TRAIN.NWORKERS = 4
    config.MODEL.TRAIN.DROPOUT = 0.2
    config.MODEL.TRAIN.CRITERION = 'MSELoss'
    config.MODEL.TRAIN.OPTIMIZER = 'Adam'
    config.MODEL.TRAIN.LEARNINGRATE = 1e-4
    config.MODEL.TRAIN.WEIGHTDECAY = 0
    config.MODEL.TRAIN.EARLYSTOP.PATIENCE = 7
    config.MODEL.TRAIN.EARLYSTOP.VERBOSE = True
    config.MODEL.TRAIN.EARLYSTOP.DELTA = 0
    config.MODEL.TRAIN.EPOCHS = 100
    config.MODEL.ARCHITECTURE = [
        256,
        'BN',
        'ReLU',
        'DO',
    ]

    return config
