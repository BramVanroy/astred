from functools import partial

import numpy as np
import pandas as pd
from tqdm import tqdm

from sac import SAC
from astred import ASTrED, GenericTree
from sac.utils import load_nlp


def process_astred_cross(method, r):
    # if the column n_src_sents and n_tgt_sents exists, they must be 1,
    # otherwise skip. A segment that contains more than one sentence is hard
    # to parse so we don't want those
    if 'n_src_sents' in r.index and 'n_tgt_sents' in r.index:
        if r['n_src_sents'] > 1 or r['n_tgt_sents'] > 1:
            return None

    # res is a dict with astred values and n_avg_tokens
    sac = SAC(**r)
    astred = ASTrED()
    res = astred.calculate(r['src_segment'], r['tgt_segment'], r['alignments'], method=method, verbose=0, draw=False)

    res['cross'] = sac.cross
    res['seq_cross'] = sac.seq_cross
    res['sac_cross'] = sac.sac_cross
    res['n_null_aligns'] = sac.n_null_aligns

    return res


def fill_nan(row_list):
    """ Get the first not-None value that we find, and use it as a template
        to fill nan-values in, for all None items.
    :param row_list: existing row_list with 'None' items
    :return: modified row_list with 'None' items replaced by dicts with nan-values
    """
    example = None
    for i in row_list:
        if i is not None:
            example = i
            break

    nan_d = {k: np.nan for k in example.keys()}
    row_list = [i if i is not None else nan_d for i in row_list]

    return row_list


def main(fin, fout, method):
    """
    :param fin: input file dataframe
    :param fout: output file
    :param method: method(s) to use
    :return:
    """
    df = pd.read_csv(fin, sep='\t')
    GenericTree.init_nlp('en', load_nlp('en'))
    GenericTree.init_nlp('nl', load_nlp('nl'))
    tqdm.pandas()

    # astred and cross
    print('ADDING CROSS AND ASTRED')
    partial_func = partial(process_astred_cross, method)

    # dirty looking way of fast processing
    # as taken from https://stackoverflow.com/a/60056244/1150683
    row_list = df.progress_apply(partial_func, axis=1)
    row_list = fill_nan(row_list)

    data_f = pd.DataFrame(row_list)

    result = pd.concat([df, data_f], axis=1, sort=False)
    result.to_csv(fout, sep='\t', index=False)


if __name__ == '__main__':
    import argparse

    cparser = argparse.ArgumentParser(description="Adds values to existing dataframe, particularly seq_alignments,"
                                                  " seq_cross, astred, and so on")

    cparser.add_argument('fin', help='Path to input file')
    cparser.add_argument('fout', help='Path to output file')
    cparser.add_argument('-m', '--method', nargs='+', choices=['default', 'token_avg', 'overlap'], default='default',
                         help='How to calculate the number of label changes.')
    cargs = cparser.parse_args()

    main(**vars(cargs))
    print(GenericTree.NLPS)