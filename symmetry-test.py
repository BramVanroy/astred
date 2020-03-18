from astred import SACr
from astred.utils import aligns_from_str, AlignmentPair
import pandas as pd

from multiprocessing import Pool
from tqdm import tqdm
import numpy as np


def process(df):
    df.apply(main, axis=1)

def main(r):
    aligns = aligns_from_str(r['alignments'])
    reverse_aligns = [AlignmentPair(t[1], t[0]) for t in aligns]

    sacr = SACr.from_list(aligns,
                        r['src_segment'],
                        r['tgt_segment'],
                        use_gpu=False)
    reverse_sacr = SACr.from_list(reverse_aligns,
                                r['tgt_segment'],
                                r['src_segment'],
                                src_lang='nl',
                                tgt_lang='en',
                                use_gpu=False)
    print('ALIGNS', sacr.aligns)
    print('SRC', sacr.src_segment)
    print('TGT', sacr.tgt_segment)
    print('CROSS', sacr.cross)
    assert sacr.cross == reverse_sacr.cross
    print('SEQ', sacr.seq_cross, reverse_sacr.seq_cross)

    print('SEQ GROUPS', sacr.seq_groups)
    assert sacr.seq_cross == reverse_sacr.seq_cross
    print('SACr', sacr.sacr_cross, reverse_sacr.sacr_cross)
    print('SACr groups', sacr.sacr_groups)
    assert sacr.sacr_cross == reverse_sacr.sacr_cross
    print()


if __name__ == '__main__':
    import argparse

    cparser = argparse.ArgumentParser(description="Check if cross, seq_cross, and sac_cross values are symmetrical")

    cparser.add_argument('fin', help='Path to input file')
    cparser.add_argument('-j', type=int, help='Number of processes to use', default=1)
    cparser.add_argument('--splits', type=int, help='Number of splits to divide the DF in', default=1)
    cargs = cparser.parse_args()

    df = pd.read_csv(cargs.fin, sep='\t')

    if cargs.j > 1:
        data_split = np.array_split(df, cargs.splits)
        with Pool(processes=cargs.j) as p:
            with tqdm(total=cargs.splits) as pbar:
                for i, _ in enumerate(p.imap(process, data_split)):
                    pbar.update()
    else:
        process(df)