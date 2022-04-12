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


set_logger()
loader = AudioFileLoader()

dict_ = {}
for prefix in [
    "timit",
    "libri",
    "clarity",
    "wsj0",
    "vctk",
]:
    spks = loader.get_speakers(prefix)
    dict_[prefix] = {}
    dict_[prefix]['speakers'] = len(spks)
    utts = sum(len(x) for x in spks.values())
    dict_[prefix]['utterances'] = utts
    dict_[prefix]['min_utt/spk'] = min(len(x) for x in spks.values())
    dict_[prefix]['max_utt/spk'] = max(len(x) for x in spks.values())
    duration = loader.get_duration(f'{prefix}_.*')[0]
    dict_[prefix]['total_length'] = f"{duration/3600:.1f}h"
    dict_[prefix]['avg_utt_length'] = f"{duration/utts:.1f}s"

pretty_table(dict_, "corpus")
