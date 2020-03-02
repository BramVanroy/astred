from functools import partial
from multiprocessing.pool import Pool

import numpy as np
import pandas as pd
from tqdm import tqdm

from sac import ASTrED


def process_astred_cross(r, use_gpu=True):
    # if the column n_src_sents and n_tgt_sents exists, they must be 1,
    # otherwise skip. A segment that contains more than one sentence is hard
    # to parse so we don't want those
    if 'n_src_sents' in r.index and 'n_tgt_sents' in r.index:
        if r['n_src_sents'] > 1 or r['n_tgt_sents'] > 1:
            return None

    # res is a dict with astred values and n_avg_tokens
    astred = ASTrED(r['src_segment'], r['tgt_segment'], r['alignments'], use_gpu=use_gpu)
    astred_mwe = ASTrED(r['src_segment'], r['tgt_segment'], r['alignments'],
                        group_mwe=True,
                        use_gpu=use_gpu)
    res = {
        'cross': astred.cross,
        'seq_cross': astred.seq_cross,
        'seq_cross_mwe': astred_mwe.seq_cross,
        'sac_cross': astred.sac_cross,
        'sac_cross_mwe': astred_mwe.sac_cross,
        'n_src_tokens': astred.n_src_tokens,
        'n_tgt_tokens': astred.n_tgt_tokens,
        'n_null_aligns': astred.n_null_aligns,
        'ted': astred.ted[0],  # regular tree edit distance
        'astred': astred.astred[0]  # tree edit distance using the merged trees
    }

    for change_type, change_d in astred.label_changes.items():
        # default has only one value, whereas avg_token and overlap
        # have a value for src and tgt
        if isinstance(change_d, dict):
            for direction, n_changes in change_d.items():
                res[f"{change_type}_changes_{direction}"] = n_changes
        else:
            res[f"{change_type}_changes"] = change_d

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


def mapable_df_process(func, df):
    return df.apply(func, axis=1)


def main(args):
    """
    :param fin: input file dataframe
    :param fout: output file
    :return:
    """
    df = pd.read_csv(args.fin, sep='\t')

    gpu_partial_process = partial(process_astred_cross, use_gpu=not args.no_cuda)

    if args.j > 1:
        splits = np.array_split(df, args.splits)
        partial_func = partial(mapable_df_process, gpu_partial_process)
        with Pool(processes=cargs.j) as pool:
            row_list = pd.concat(tqdm(pool.imap(partial_func, splits),
                                      total=args.splits,
                                      unit='split'))
    else:
        tqdm.pandas(unit='row')
        row_list = df.progress_apply(gpu_partial_process, axis=1)

    row_list = fill_nan(row_list)
    data_f = pd.DataFrame(row_list)
    result = pd.concat([df, data_f], axis=1, sort=False)
    result.to_csv(cargs.fout, sep='\t', index=False)


if __name__ == '__main__':
    import argparse

    cparser = argparse.ArgumentParser(
        description="Adds cross-related values to an existing DataFrame. DataFrame must contain"
                    " columns 'src_segment', 'tgt_segment' and 'alignments'")

    cparser.add_argument('fin', help='Path to input file')
    cparser.add_argument('fout', help='Path to output file')
    cparser.add_argument('-j', help='Processes to use', type=int, default=1)
    cparser.add_argument('--splits', help='Number of splits to divide the dataset in', type=int,
                         default=1)
    cparser.add_argument('--no_cuda', help="Disable CUDA. Useful when using multiple processes,"
                                           " otherwise the multiple processes' models might not"
                                           " fit on GPU",
                         action='store_true',
                         default=False)
    cargs = cparser.parse_args()

    if cargs.splits < cargs.j:
        raise ValueError(
            "The number of splits is smaller than the number of processes (-j), which doesn't"
            " make sense. Please ensure that the number of splits is greater than or equal"
            " to the number of processes")

    main(cargs)
