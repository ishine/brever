import pytest

from brever.io import AudioFileLoader


def test_errors():
    loader = AudioFileLoader()
    with pytest.raises(ValueError):
        loader.get_speech_files('dcase_bus')
    with pytest.raises(ValueError):
        loader.get_noise_files('libri_.*')
    with pytest.raises(ValueError):
        loader.get_angles('noisex_babble')


def test_speakers():
    loader = AudioFileLoader()
    speakers = loader.get_speakers('timit')
    assert len(speakers) == 630
    speakers = loader.get_speakers('libri')
    assert len(speakers) == 251


def test_target():
    file_count = {
        'timit_.*': 6300,
        'timit_m0': 10,
        'timit_f0': 10,
        'timit_m1': 10,
        'timit_f1': 10,
        'timit_m[0-4]': 50,
        'timit_f[0-4]': 50,
        'timit_(f[0-4]|m[0-4])': 100,
        'timit_(f[0-9]?[02468]|m[0-9]?[02468])': 1000,
        'libri_.*': 28539,
        'libri_m0': 118,
        'libri_f0': 111,
        'libri_m1': 138,
        'libri_f1': 117,
        'libri_m[0-4]': 608,
        'libri_f[0-4]': 588,
        'libri_(f[0-4]|m[0-4])': 1196,
        'libri_(f[0-9]?[02468]|m[0-9]?[02468])': 11454,
        'ieee': 720,
        'arctic': 4528,
        'hint': 260,
    }
    loader = AudioFileLoader()
    for speaker in file_count.keys():
        files = loader.get_speech_files(speaker)
        assert len(files) == file_count[speaker]


def test_noises():
    noises = [
        'dcase_.*',
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
        'noisex_.*',
        'noisex_babble',
        'noisex_buccaneer1',
        'noisex_buccaneer2',
        'noisex_destroyerengine',
        'noisex_destroyerops',
        'noisex_f16',
        'noisex_factory1',
        'noisex_factory2',
        'noisex_hfchannel',
        'noisex_leopard',
        'noisex_m109',
        'noisex_machinegun',
        'noisex_pink',
        'noisex_volvo',
        'noisex_white',
        'icra_.*',
        'icra_01',
        'icra_02',
        'icra_03',
        'icra_04',
        'icra_05',
        'icra_06',
        'icra_07',
        'icra_08',
        'icra_09',
        'demand',
        'arte',
    ]
    loader = AudioFileLoader()
    for noise in noises:
        loader.get_noise_files(noise)


@pytest.mark.slow
def test_brirs():
    file_count = {
        'surrey_anechoic': 37,
        'surrey_room_a': 37,
        'surrey_room_b': 37,
        'surrey_room_c': 37,
        'surrey_room_d': 37,
        'ash_r01': 24,
        'ash_r02': 24,
        'ash_r03': 24,
        'ash_r04': 24,
        'ash_r05a': 24,
        'ash_r05b': 20,
        'ash_r06': 20,
        'ash_r07': 24,
        'ash_r08': 9,
        'ash_r09': 9,
        'ash_r10': 18,
        'ash_r11': 18,
        'ash_r12': 9,
        'ash_r13': 9,
        'ash_r14': 9,
        'ash_r15': 9,
        'ash_r16': 18,
        'ash_r17': 10,
        'ash_r18': 16,
        'ash_r19': 7,
        'ash_r20': 5,
        'ash_r21': 7,
        'ash_r22': 11,
        'ash_r23': 16,
        'ash_r24': 16,
        'ash_r25': 14,
        'ash_r26': 16,
        'ash_r27': 5,
        'ash_r28': 14,
        'ash_r29': 14,
        'ash_r30': 14,
        'ash_r31': 14,
        'ash_r32': 7,
        'ash_r33': 14,
        'ash_r34': 14,
        'ash_r35': 14,
        'ash_r36': 14,
        'ash_r37': 14,
        'ash_r38': 14,
        'ash_r39': 14,
        'air_aula_carolina_1_1': 1,
        'air_aula_carolina_1_2': 1,
        'air_aula_carolina_1_3': 5,
        'air_aula_carolina_1_4': 1,
        'air_aula_carolina_1_5': 1,
        'air_aula_carolina_1_6': 1,
        'air_aula_carolina_1_7': 1,
        'air_booth_0_1': 1,
        'air_booth_0_2': 1,
        'air_booth_0_3': 1,
        'air_booth_1_1': 1,
        'air_booth_1_2': 1,
        'air_booth_1_3': 1,
        'air_lecture_0_1': 1,
        'air_lecture_0_2': 1,
        'air_lecture_0_3': 1,
        'air_lecture_0_4': 1,
        'air_lecture_0_5': 1,
        'air_lecture_0_6': 1,
        'air_lecture_1_1': 1,
        'air_lecture_1_2': 1,
        'air_lecture_1_3': 1,
        'air_lecture_1_4': 1,
        'air_lecture_1_5': 1,
        'air_lecture_1_6': 1,
        'air_meeting_0_1': 1,
        'air_meeting_0_2': 1,
        'air_meeting_0_3': 1,
        'air_meeting_0_4': 1,
        'air_meeting_0_5': 1,
        'air_meeting_1_1': 1,
        'air_meeting_1_2': 1,
        'air_meeting_1_3': 1,
        'air_meeting_1_4': 1,
        'air_meeting_1_5': 1,
        'air_office_0_1': 1,
        'air_office_0_2': 1,
        'air_office_0_3': 1,
        'air_office_1_1': 1,
        'air_office_1_2': 1,
        'air_office_1_3': 1,
        'air_stairway_1_1': 13,
        'air_stairway_1_2': 13,
        'air_stairway_1_3': 13,
        'catt_00': 37,
        'catt_01': 37,
        'catt_02': 37,
        'catt_03': 37,
        'catt_04': 37,
        'catt_05': 37,
        'catt_06': 37,
        'catt_07': 37,
        'catt_08': 37,
        'catt_09': 37,
        'catt_10': 37,
        'avil_anechoic': 24,
        'avil_high': 24,
        'avil_low': 24,
        'avil_medium': 24,
        'elospheres_anechoic': 73,
        'elospheres_restaurant': 73,
        'elospheres_kitchen': 73,
        'bras_cr2': 45,
        'bras_cr3': 45,
        'bras_cr4': 45,
        'bras_rs5': 45,
        'surrey_.*': 185,
        'ash_.*': 576,
        'air_.*': 84,
        'catt_.*': 407,
        'avil_.*': 96,
        'elospheres_.*': 219,
        'bras_.*': 180,
    }
    loader = AudioFileLoader()
    for regexp in file_count.keys():
        rooms = loader.get_rooms(regexp)
        n = 0
        for room in rooms:
            brirs, _ = loader.load_brirs(room)
            n += len(brirs)
        assert n == file_count[regexp]
