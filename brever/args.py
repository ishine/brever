import argparse


def arg_type_bool(x):
    return bool(int(x))


def arg_type_path(x):
    return x.replace('\\', '/').rstrip('/')


class SetAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, set(values))


class DatasetArgParser(argparse.ArgumentParser):

    arg_map = {
        # general
        'duration': ['DURATION'],
        'seed': ['SEED'],
        'speakers': ['SPEAKERS'],
        'noises': ['NOISES'],
        'rooms': ['ROOMS'],
        'padding': ['PADDING'],
        'reflection_boundary': ['REFLECTION_BOUNDARY'],
        'uniform_tmr': ['UNIFORM_TMR'],
        'components': ['COMPONENTS'],

        # target
        'snr_dist_name': ['TARGET', 'SNR', 'DIST_NAME'],
        'snr_dist_args': ['TARGET', 'SNR', 'DIST_ARGS'],
        'target_angle': ['TARGET', 'ANGLE'],

        # noise
        'noise_num': ['NOISE', 'NUMBER'],
        'noise_angle': ['NOISE', 'ANGLE'],
        'ndr_dist_name': ['NOISE', 'NDR', 'DIST_NAME'],
        'ndr_dist_args': ['NOISE', 'NDR', 'DIST_ARGS'],

        # diffuse noise
        'diffuse': ['DIFFUSE', 'TOGGLE'],
        'diffuse_color': ['DIFFUSE', 'COLOR'],
        'ltas_eq': ['DIFFUSE', 'LTAS_EQ'],

        # decay
        'decay': ['DECAY', 'TOGGLE'],
        'decay_color': ['DECAY', 'COLOR'],
        'rt60_dist_name': ['DECAY', 'RT60', 'DIST_NAME'],
        'rt60_dist_args': ['DECAY', 'RT60', 'DIST_ARGS'],
        'delay_dist_name': ['DECAY', 'DELAY', 'DIST_NAME'],
        'delay_dist_args': ['DECAY', 'DELAY', 'DIST_ARGS'],
        'drr_dist_name': ['DECAY', 'DRR', 'DIST_NAME'],
        'drr_dist_args': ['DECAY', 'DRR', 'DIST_ARGS'],

        # rms jitter
        'rms_jitter_dist_name': ['RMS_JITTER', 'DIST_NAME'],
        'rms_jitter_dist_args': ['RMS_JITTER', 'DIST_ARGS'],

        # file limits
        'speech_files': ['FILES', 'SPEECH'],
        'noise_files': ['FILES', 'NOISE'],
        'room_files': ['FILES', 'ROOM'],
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        group = self.add_argument_group('general options')
        group.add_argument('--duration', type=int)
        group.add_argument('--seed', type=int)
        group.add_argument('--speakers', nargs='+', action=SetAction)
        group.add_argument('--noises', nargs='+', action=SetAction)
        group.add_argument('--rooms', nargs='+', action=SetAction)
        group.add_argument('--padding', type=float)
        group.add_argument('--reflection-boundary', type=float)
        group.add_argument('--uniform-tmr', type=arg_type_bool)
        group.add_argument('--components', nargs='+', action=SetAction)

        group = self.add_argument_group('target options')
        group.add_argument('--snr-dist-name')
        group.add_argument('--snr-dist-args', type=float, nargs=2)
        group.add_argument('--target-angle', type=float, nargs=2)

        group = self.add_argument_group('noise options')
        group.add_argument('--noise-num', type=int, nargs=2)
        group.add_argument('--noise-angle', type=float, nargs=2)
        group.add_argument('--ndr-dist-name')
        group.add_argument('--ndr-dist-args', type=float, nargs=2)

        group = self.add_argument_group('diffuse noise options')
        group.add_argument('--diffuse', type=arg_type_bool)
        group.add_argument('--diffuse-color')
        group.add_argument('--ltas-eq', type=arg_type_bool)

        group = self.add_argument_group('decay options')
        group.add_argument('--decay', type=arg_type_bool)
        group.add_argument('--decay-color')
        group.add_argument('--rt60-dist-name')
        group.add_argument('--rt60-dist-args', type=float, nargs=2)
        group.add_argument('--delay-dist-name')
        group.add_argument('--delay-dist-args', type=float, nargs=2)
        group.add_argument('--drr-dist-name')
        group.add_argument('--drr-dist-args', type=float, nargs=2)

        group = self.add_argument_group('rms jitter options')
        group.add_argument('--rms-jitter-dist-name')
        group.add_argument('--rms-jitter-dist-args', type=float, nargs=2)

        group = self.add_argument_group('file limit options')
        group.add_argument('--speech-files', type=float, nargs=2)
        group.add_argument('--noise-files', type=float, nargs=2)
        group.add_argument('--room-files')


class TrainingArgParser(argparse.ArgumentParser):

    arg_map = {
        'batch_size': ['BATCH_SIZE'],
        'cuda': ['CUDA'],
        'early_stop': ['EARLY_STOP', 'TOGGLE'],
        'convergence': ['CONVERGENCE', 'TOGGLE'],
        'epochs': ['EPOCHS'],
        'learning_rate': ['LEARNING_RATE'],
        'workers': ['WORKERS'],
        'weight_decay': ['WEIGHT_DECAY'],
        'train_path': ['PATH'],
        'seed': ['SEED'],
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._add_arguments()

    def _add_arguments(self):
        self.add_argument('--batch-size', type=int)
        self.add_argument('--cuda', type=arg_type_bool)
        self.add_argument('--early-stop', type=arg_type_bool)
        self.add_argument('--convergence', type=arg_type_bool)
        self.add_argument('--epochs', type=int)
        self.add_argument('--learning-rate', type=float)
        self.add_argument('--workers', type=int)
        self.add_argument('--weight-decay', type=float)
        self.add_argument('--train-path', type=arg_type_path)
        self.add_argument('--seed', type=int)


class ModelArgParser(argparse.ArgumentParser):

    arg_map = {
        'dnn': {
            'batch_norm': ['BATCH_NORM', 'TOGGLE'],
            'dropout': ['DROPOUT'],
            'hidden_layers': ['HIDDEN_LAYERS'],
            'online_norm': ['NORMALIZATION', 'ONLINE'],
            'group_norm': ['NORMALIZATION', 'GROUP'],
            'features': ['FEATURES'],
            'decimation': ['DECIMATION'],
            'dct_coeff': ['DCT_COEFF'],
            'stacks': ['STACKS'],
            'scale_rms': ['SCALE_RMS'],
        },
        'convtasnet': {
            'filters': ['ENCODER', 'FILTERS'],
            'filter_length': ['ENCODER', 'FILTER_LENGTH'],
            'bottleneck_channels': ['TCN', 'BOTTLENECK_CHANNELS'],
            'hidden_channels': ['TCN', 'HIDDEN_CHANNELS'],
            'skip_channels': ['TCN', 'SKIP_CHANNELS'],
            'kernel_size': ['TCN', 'KERNEL_SIZE'],
            'layers': ['TCN', 'LAYERS'],
            'repeats': ['TCN', 'REPEATS'],
            'sources': ['TCN', 'SOURCES'],
        },
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        subs = self.add_subparsers(
            help='architecture selection',
            dest='arch',
            parser_class=argparse.ArgumentParser,
            required=True,
        )

        sub = subs.add_parser('dnn')
        sub.add_argument('--batch-norm', type=arg_type_bool)
        sub.add_argument('--dropout', type=float)
        sub.add_argument('--hidden-layers', type=int, nargs='+')
        sub.add_argument('--online-norm', type=arg_type_bool)
        sub.add_argument('--group-norm', type=arg_type_bool)
        sub.add_argument('--features', nargs='+', action=SetAction)
        sub.add_argument('--decimation', type=int)
        sub.add_argument('--dct-coeff', type=int)
        sub.add_argument('--stacks', type=int)
        sub.add_argument('--scale-rms', type=arg_type_bool)

        sub = subs.add_parser('convtasnet')
        sub.add_argument('--filters', type=int)
        sub.add_argument('--filter-length', type=int)
        sub.add_argument('--bottleneck-channels', type=int)
        sub.add_argument('--hidden-channels', type=int)
        sub.add_argument('--skip-channels', type=int)
        sub.add_argument('--kernel-size', type=int)
        sub.add_argument('--layers', type=int)
        sub.add_argument('--repeats', type=int)
        sub.add_argument('--sources', type=int)
