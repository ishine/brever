import sys
import argparse
import os

import brever.modelmanagement as bmm


def find_dset(
            dsets=None,
            kind=None,
            speakers={'ieee'},
            rooms={'surrey_room_a'},
            snr_dist_args=[0, 0],
            target_angle_min=0.0,
            target_angle_max=0.0,
            noise_types={'dcase_airport'},
            random_rms=False,
        ):
    return bmm.find_dataset(
        dsets=dsets,
        kind=kind,
        speakers=speakers,
        rooms=rooms,
        snr_dist_args=snr_dist_args,
        target_angle_min=target_angle_min,
        target_angle_max=target_angle_max,
        noise_types=noise_types,
        random_rms=random_rms,
    )


class Logger:
    def __init__(self, filename, end='\n'):
        self.file = open(filename, 'w')
        self.end = end

    def write(self, data):
        self.file.write(data + self.end)
        self.file.flush()
        sys.stdout.write(data + self.end)
        sys.stdout.flush()


def write_exp(
            dim,
            model_dim_vals,
            model_labels,
            cond_dim_vals,
            cond_labels,
            filename,
            output_dir,
            rotation=None,
            lw=None,
        ):
    assert len(model_dim_vals) == len(model_labels)
    assert len(cond_dim_vals) == len(cond_labels)
    if isinstance(dim, str):
        dim = [dim]
        model_dim_vals = [(val, ) for val in model_dim_vals]
        cond_dim_vals = [(val, ) for val in cond_dim_vals]
    else:
        assert all(len(val) == len(dim) for val in model_dim_vals)
        assert all(len(val) == len(dim) for val in cond_dim_vals)
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    logger = Logger(filename, end=' \\\n')
    logger.write('python scripts/compare_models.py -i')
    pre_dsets = find_dset(
        kind='train',
        **{dim: None for dim in dim},
    )
    for val in model_dim_vals:
        dsets = find_dset(
            dsets=pre_dsets,
            **{dim: val for dim, val in zip(dim, val)},
        )
        assert len(dsets) == 1
        for dset in dsets:
            if args.normalization == 'both':
                normalizations = ['global', 'recursive']
            else:
                normalizations = [args.normalization]
            for normalization in normalizations:
                models = bmm.find_model(
                    train_path=[dset],
                    dropout=[True],
                    normalization=normalization,
                )
                assert len(models) == 1
                for model in models:
                    logger.write(model)
    logger.write('-t')
    pre_dsets = find_dset(
        kind='test',
        **{dim: None for dim in dim},
    )
    for val in cond_dim_vals:
        dsets = find_dset(
            dsets=pre_dsets,
            **{dim: val for dim, val in zip(dim, val)},
        )
        assert len(dsets) == 1
        for dset in dsets:
            logger.write(dset)
    logger.write('--legend')
    for label in model_labels:
        if args.normalization == 'both':
            for normalization in ['global', 'recursive']:
                logger.write(f'"{label} - {normalization}"')
        else:
            logger.write(f'"{label}"')
    logger.write('--xticks')
    for label in cond_labels:
        logger.write(f'"{label}"')
    logger.write('--train-curve')
    logger.write('--output-dir')
    logger.write(output_dir)
    if rotation is not None:
        logger.write('--rotation')
        logger.write(rotation)
    if lw is not None:
        logger.write('--lw')
        logger.write(lw)
    if args.normalization == 'both':
        logger.write('--group-by')
        logger.write('train-path')
    logger.write('--dims')
    logger.write('normalization')
    logger.write('train-path')
    logger.write('--summary')


def main():
    write_exp(
        dim='noise_types',
        model_dim_vals=[
            {'dcase_airport'},
            {'dcase_bus'},
            {'dcase_metro'},
            {'dcase_metro_station'},
            {'dcase_park'},
            {'dcase_public_square'},
            {'dcase_shopping_mall'},
            {'dcase_street_pedestrian'},
            {'dcase_street_traffic'},
            {'dcase_tram'},
            {
                'dcase_airport',
                'dcase_bus',
                'dcase_metro',
                'dcase_metro_station',
                'dcase_park',
                'dcase_public_square',
                'dcase_shopping_mall',
                'dcase_street_pedestrian',
                'dcase_street_traffic',
                'dcase_tram',
            },
        ],
        model_labels=[
            'airport',
            'bus',
            'metro',
            'metro_station',
            'park',
            'public_square',
            'shopping_mall',
            'street_pedestrian',
            'street_traffic',
            'tram',
            'general',
        ],
        cond_dim_vals=[
            {'dcase_airport'},
            {'dcase_bus'},
            {'dcase_metro'},
            {'dcase_metro_station'},
            {'dcase_park'},
            {'dcase_public_square'},
            {'dcase_shopping_mall'},
            {'dcase_street_pedestrian'},
            {'dcase_street_traffic'},
            {'dcase_tram'},
        ],
        cond_labels=[
            'airport',
            'bus',
            'metro',
            'metro_station',
            'park',
            'public_square',
            'shopping_mall',
            'street_pedestrian',
            'street_traffic',
            'tram',
            ],
        filename=f'experiments/noise_{args.normalization}.sh',
        output_dir='pics/exp/noise',
        rotation='45',
        lw='0.4',
    )
    write_exp(
        dim='speakers',
        model_dim_vals=[
            {'ieee'},
            {'timit_.*'},
            {'timit_FCJF0'},
            {'timit_^(?!FCJF0$).*$'},
            {'libri_.*'},
            {'libri_19'},
            {'libri_^(?!19$).*$'},
            {'timit_.*', 'libri_.*'},
        ],
        model_labels=[
            'ieee',
            'timit_all',
            'timit_FCJF0',
            'timit_all_but_FCJF0',
            'libri_all',
            'libri_19',
            'libri_all_but_19',
            'timit_all and libri_all',
        ],
        cond_dim_vals=[
            {'ieee'},
            {'timit_.*'},
            {'timit_FCJF0'},
            {'timit_^(?!FCJF0$).*$'},
            {'libri_.*'},
            {'libri_19'},
            {'libri_^(?!19$).*$'},
        ],
        cond_labels=[
            'ieee',
            'timit_all',
            'timit_FCJF0',
            'timit_all_but_FCJF0',
            'libri_all',
            'libri_19',
            'libri_all_but_19',
        ],
        filename=f'experiments/speaker_{args.normalization}.sh',
        output_dir='pics/exp/speaker',
    )
    write_exp(
        dim='snr_dist_args',
        model_dim_vals=[
            [-5, -5],
            [0, 0],
            [5, 5],
            [10, 10],
            [-5, 10],
        ],
        model_labels=[
            '-5 dB SNR',
            '0 dB SNR',
            '5 dB SNR',
            '10 dB SNR',
            '-5 -- 10 dB SNR',
        ],
        cond_dim_vals=[
            [-5, -5],
            [0, 0],
            [5, 5],
            [10, 10],
        ],
        cond_labels=[
            '-5 dB SNR',
            '0 dB SNR',
            '5 dB SNR',
            '10 dB SNR',
        ],
        filename=f'experiments/snr_{args.normalization}.sh',
        output_dir='pics/exp/snr',
    )
    write_exp(
        dim='rooms',
        model_dim_vals=[
            {'surrey_room_a'},
            {'surrey_room_b'},
            {'surrey_room_c'},
            {'surrey_room_d'},
            {'surrey_room_.'},
            {'ash_r01'},
            {'ash_r0.*'},
            {'^ash_r(?!0).*$'},
            {'^ash_r(?!01$).*$'},
            {'ash_r.*'},
            {'surrey_room_.', 'ash_r.*'},
        ],
        model_labels=[
            'SURREY A',
            'SURREY B',
            'SURREY C',
            'SURREY D',
            'SURREY all',
            'ASH 01',
            'ASH 01-09',
            'ASH all but 01-09',
            'ASH all but 01',
            'ASH all',
            'SURREY all and ASH all',
        ],
        cond_dim_vals=[
            {'surrey_room_a'},
            {'surrey_room_b'},
            {'surrey_room_c'},
            {'surrey_room_d'},
            {'surrey_room_.'},
            {'ash_r01'},
            {'ash_r0.*'},
            {'^ash_r(?!0).*$'},
            {'^ash_r(?!01$).*$'},
            {'ash_r.*'},
        ],
        cond_labels=[
            'SURREY A',
            'SURREY B',
            'SURREY C',
            'SURREY D',
            'SURREY all',
            'ASH 01',
            'ASH 01-09',
            'ASH all but 01-09',
            'ASH all but 01',
            'ASH all',
        ],
        filename=f'experiments/room_{args.normalization}.sh',
        output_dir='pics/exp/room',
    )
    write_exp(
        dim=('target_angle_min', 'target_angle_max'),
        model_dim_vals=[
            (0, 0),
            (-90, 90),
        ],
        model_labels=[
            'fixed speaker location',
            'random speaker location',
        ],
        cond_dim_vals=[
            (0, 0),
            (-90, 90),
        ],
        cond_labels=[
            'fixed speaker location',
            'random speaker location',
        ],
        filename=f'experiments/angle_{args.normalization}.sh',
        output_dir='pics/exp/angle',
    )
    write_exp(
        dim='random_rms',
        model_dim_vals=[
            False,
            True,
        ],
        model_labels=[
            'fixed speaker level',
            'random mixture level',
        ],
        cond_dim_vals=[
            False,
            True,
        ],
        cond_labels=[
            'fixed speaker level',
            'random mixture level',
        ],
        filename=f'experiments/rms_{args.normalization}.sh',
        output_dir='pics/exp/rms',
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--normalization', default='global')
    args = parser.parse_args()
    main()
