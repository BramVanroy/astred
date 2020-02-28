from sac import aligns_from_str, SAC
import pandas as pd

from tqdm import tqdm

def main(r):
    aligns = aligns_from_str(r['alignments'])
    reverse_aligns = [(t[1], t[0]) for t in aligns]

    sac = SAC.from_list(aligns,
                        r['src_segment'],
                        r['tgt_segment'])
    reverse_sac = SAC.from_list(reverse_aligns,
                                r['tgt_segment'],
                                r['src_segment'],
                                src_lang='nl',
                                tgt_lang='en')
    print('ALIGNS', sac.aligns)
    print('SRC', sac.src_text)
    print('TGT', sac.tgt_text)
    assert sac.cross == reverse_sac.cross
    print('CROSS', sac.cross)
    assert sac.seq_cross == reverse_sac.seq_cross
    print('SEQ CROSS', sac.seq_cross)
    print('SEQ GROUPS', sac.seq_groups)
    assert sac.sac_cross == reverse_sac.sac_cross
    print('SAC', sac.sac_cross)
    print('SAC groups', sac.sac_groups)
    print()


if __name__ == '__main__':
    import argparse

    cparser = argparse.ArgumentParser(description="Check if cross, seq_cross, and sac_cross values are symmetrical")

    cparser.add_argument('fin', help='Path to input file')
    cargs = cparser.parse_args()
    df = pd.read_csv(cargs.fin, sep='\t')
    tqdm.pandas()
    df.progress_apply(main, axis=1)
