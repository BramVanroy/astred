from copy import deepcopy

import stanfordnlp
from nltk.draw import TreeView
from spacy_stanfordnlp import StanfordNLPLanguage
from stanfordnlp.utils.resources import DEFAULT_MODEL_DIR


def draw_trees(*trees, include_word_idx=False):
    """
    Open a new window containing a graphical diagram of the given
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


def load_nlp(lang_or_model, tokenize_pretokenized=True, use_gpu=True):
    stanfordnlp.download(lang_or_model, DEFAULT_MODEL_DIR)
    snlp = stanfordnlp.Pipeline(
        processors="tokenize,pos,depparse",
        lang=lang_or_model,
        tokenize_pretokenized=tokenize_pretokenized,
        use_gpu=use_gpu,
    )
    return StanfordNLPLanguage(snlp)


def aligns_from_str(aligns):
    return sorted([tuple(map(int, align.split("-"))) for align in aligns.split()])


def aligns_to_str(aligns):
    """ Convert list of alignments (tuple of src, tgt) to GIZA/Pharaoh string """
    return " ".join([f"{s}-{t}" for s, t in aligns])
