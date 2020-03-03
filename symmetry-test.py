from sacr import aligns_from_str, SAC
import pandas as pd

from multiprocessing import Pool
from tqdm import tqdm
import numpy as np


def process(df):
    df.apply(main, axis=1)

def main(r):
    aligns = aligns_from_str(r['alignments'])
    reverse_aligns = [(t[1], t[0]) for t in aligns]

    sac = SAC.from_list(aligns,
                        r['src_segment'],
                        r['tgt_segment'],
                        use_gpu=False)
    reverse_sac = SAC.from_list(reverse_aligns,
                                r['tgt_segment'],
                                r['src_segment'],
                                src_lang='nl',
                                tgt_lang='en',
                                use_gpu=False)
    # print('ALIGNS', sac.aligns)
    # print('SRC', sac.src_text)
    # print('TGT', sac.tgt_text)
    # print('CROSS', sac.cross)
    assert sac.cross == reverse_sac.cross
    # print('SEQ', sac.seq_cross, reverse_sac.seq_cross)
    # print('SEQ GROUPS', sac.seq_groups)
    assert sac.seq_cross == reverse_sac.seq_cross
    # print('SAC', sac.sac_cross, reverse_sac.sac_cross)
    # print('SAC groups', sac.sac_groups)
    assert sac.sac_cross == reverse_sac.sac_cross
    # print()


if __name__ == '__main__':
    import argparse

    cparser = argparse.ArgumentParser(description="Check if cross, seq_cross, and sac_cross values are symmetrical")

    cparser.add_argument('fin', help='Path to input file')
    cparser.add_argument('-j', type=int, help='Number of processes to use', default=1)
    cparser.add_argument('--splits', type=int, help='Number of splits to divide the DF in', default=1)
    cargs = cparser.parse_args()

    df = pd.read_csv(cargs.fin, sep='\t')
    data_split = np.array_split(df, cargs.splits)
    with Pool(processes=cargs.j) as p:
        with tqdm(total=cargs.splits) as pbar:
            for i, _ in enumerate(p.imap(process, data_split)):
                pbar.update()
