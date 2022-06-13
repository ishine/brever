from brever.io import AudioFileLoader
from brever.logger import set_logger
from brever.display import pretty_table


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
