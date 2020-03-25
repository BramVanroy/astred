""" Example showing how to use the GenericTree class and utilities to use tree edit distance
    on a monolingual task. """

from astred import GenericTree
from astred.utils import draw_trees, get_distance


def main(src, tgt, lang="en", pretokenized=False, draw=False):
    tree_ref = GenericTree.from_string(
        src, lang_or_model=lang, tokenize_pretokenized=pretokenized
    )
    tree_hyp = GenericTree.from_string(
        tgt, lang_or_model=lang, tokenize_pretokenized=pretokenized
    )

    distance_dep, operations_dep = get_distance(tree_ref, tree_hyp)
    distance_tok, operations_tok = get_distance(tree_ref.text_tree, tree_hyp.text_tree)

    print("DEPENDENCY DISTANCE", distance_dep)
    # print the operations that were used to calculate the distance
    for op in operations_dep:
        print(op)
    print()
    if draw:
        draw_trees(tree_ref, tree_hyp, include_word_idx=True)

    print("TOKEN DISTANCE", distance_tok)
    for op in operations_tok:
        print(op)
    print()
    if draw:
        draw_trees(tree_ref.text_tree, tree_hyp.text_tree, include_word_idx=True)

    # verify that the we can retrieve the correct order of the tokens from the tree
    # taking advantage of ordered dicts in 3.6+
    if pretokenized:
        assert (
            " ".join(
                subtree.label()
                for subtree in tree_ref.text_tree.word_order_idx_mapping.values()
            )
            == src
        )
        assert (
            " ".join(
                subtree.label()
                for subtree in tree_hyp.text_tree.word_order_idx_mapping.values()
            )
            == tgt
        )


if __name__ == "__main__":
    import argparse

    cparser = argparse.ArgumentParser(description=__doc__)
    cparser.add_argument("src", help="Source text")
    cparser.add_argument("tgt", help="Target text")
    cparser.add_argument("-l", "--lang", help="Language of text (language code)", default="en")
    cparser.add_argument(
        "-t",
        "--pretokenized",
        help="Whether the text is already tokenized",
        action="store_true",
    )
    cparser.add_argument(
        "-d",
        "--draw",
        help="Whether to draw the trees (NOT recommended when running code remotely)",
        action="store_true",
    )

    cargs = vars(cparser.parse_args())
    main(**cargs)
