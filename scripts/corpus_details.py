from brever.io import AudioFileLoader
from brever.logger import set_logger


def pretty_table(dict_: dict, key_header: str = '') -> None:
    if not dict_:
        raise ValueError('input is empty')
    keys = dict_.keys()
    values = dict_.values()
    first_col_width = max(max(len(str(key)) for key in keys), len(key_header))
    col_widths = [first_col_width]
    for i, value in enumerate(values):
        if i == 0:
            sub_keys = value.keys()
        elif value.keys() != sub_keys:
            raise ValueError('values in input do not all have same keys')
    for key in sub_keys:
        col_width = max(max(len(str(v[key])) for v in values), len(key))
        col_widths.append(col_width)
    row_fmt = ' '.join(f'{{:<{width}}} ' for width in col_widths)
    print(row_fmt.format(key_header, *sub_keys))
    print(row_fmt.format(*['-'*w for w in col_widths]))
    for key, items in dict_.items():
        print(row_fmt.format(key, *items.values()))


def main():
    set_logger()
    loader = AudioFileLoader()

    dict_ = {}
    for db in [
        "timit",
        "libri",
        "clarity",
        "wsj0",
        "vctk",
    ]:
        spks = loader.get_speakers(db)
        dict_[db] = {}
        dict_[db]['speakers'] = len(spks)
        utts = sum(len(x) for x in spks.values())
        utts_per_spk = [len(x) for x in spks.values()]
        dict_[db]['utterances'] = utts
        dict_[db]['avg_utt/spk'] = round(sum(utts_per_spk)/len(utts_per_spk))
        dict_[db]['min_utt/spk'] = min(utts_per_spk)
        dict_[db]['max_utt/spk'] = max(utts_per_spk)
        duration = loader.get_duration(f'{db}_.*', reduce_=False)[0]
        total = sum(duration)
        dict_[db]['duration'] = f"{total/3600:.1f}h"
        dict_[db]['avg_utt_len'] = f"{total/utts:.1f}s"
        dict_[db]['min_utt_len'] = f"{min(duration):.1f}s"
        dict_[db]['max_utt_len'] = f"{max(duration):.1f}s"

    pretty_table(dict_, "corpus")


if __name__ == '__main__':
    main()
