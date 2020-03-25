from copy import deepcopy
from typing import List, NamedTuple, Tuple, Union

import stanza
from apted import APTED, helpers
from nltk.draw import TreeView
from nltk.tree import ParentedTree

AlignmentPair = NamedTuple("AlignmentPair", [("src", int), ("tgt", int)])


def aligns_from_str(aligns: str) -> List[AlignmentPair]:
    return sorted([AlignmentPair(*map(int, align.split("-"))) for align in aligns.split()])


def aligns_to_str(aligns: List[Union[Tuple[int, int], AlignmentPair]]) -> str:
    """Convert list of alignments (tuple of src, tgt) to GIZA/Pharaoh string """
    return " ".join([f"{src}-{tgt}" for src, tgt in aligns])


def draw_trees(*trees: ParentedTree, include_word_idx: bool = False):
    """Open a new window containing a graphical diagram of the given
    trees. Optionally prepend the word index to the labels so
    that it is visually more clear which word is where in the tree

    :rtype: None
    """
    word_idxs_trees = []
    if include_word_idx:
        for tree in trees:
            tree_copy = deepcopy(tree)
            tree_copy.add_word_idx_to_label()
            word_idxs_trees.append(tree_copy)
    else:
        word_idxs_trees = trees

    TreeView(*word_idxs_trees).mainloop()


def get_distance(src_tree, tgt_tree):
    """Calculate the distance between the source and target tree.
    :return: the tree edit distance for the given trees and optionally the required operations
    """
    src_tree_str = src_tree.to_string(parens="{}")
    tree_src_apted = helpers.Tree.from_text(src_tree_str)
    tgt_tree_str = tgt_tree.to_string(parens="{}")
    tgt_tree_apted = helpers.Tree.from_text(tgt_tree_str)

    apted = APTED(tree_src_apted, tgt_tree_apted)
    dist = apted.compute_edit_distance()
    opts = apted.compute_edit_mapping()

    return dist, opts


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
