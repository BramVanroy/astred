from typing import List

import stanza


def unique_list(groups: List):
    """Filter list of lists so that:
       - the sublists only contain unique items (no duplicates);
       - the sublists themselves are unique (two identical sublists cannot exists)"""

    def unique(main_list: List):
        uniq = []
        uniq_ids = set()
        for item in main_list:
            is_list = isinstance(item, list)
            item = [item] if not is_list else item
            item_repr = tuple(
                [f"{i.doc.side if i.doc else 'none'}-{i.id}" for i in item]
            )
            if item_repr not in uniq_ids:
                uniq.append(item[0] if not is_list else item)
                uniq_ids.add(item_repr)
        return uniq

    if isinstance(groups[0], list):
        # Make sure that items in sublists are unique
        groups = [unique(group) for group in groups]

    # Make sure that sublists themselves are unique
    return unique(groups)


def rebase_to_idxs(idxs: List[int]):
    """ Convert values to indices. This ensure that there are no strange gaps
        between sequence alignments (e.g. when an index is not word-aligned) """
    l_sort = sorted(list(set(idxs)))

    return [l_sort.index(x) for x in idxs]


def pair_combs(all_pairs, min_length=2):
    n_pairs = len(all_pairs)
    for i in range(n_pairs, min_length - 1, -1):
        for j in range(n_pairs - i + 1):
            pairs = all_pairs[j : j + i]
            if any(item.is_null for pair in pairs for item in pair):
                continue
            yield pairs


def load_nlp(
    lang: str,
    tokenize_pretokenized: bool = True,
    use_gpu: bool = True,
    logging_level: str = "INFO",
):
    return stanza.Pipeline(
        processors="tokenize,mwt,pos,lemma,depparse",
        lang=lang,
        tokenize_pretokenized=tokenize_pretokenized,
        use_gpu=use_gpu,
        logging_level=logging_level,
    )
