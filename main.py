from sac import AlignedTrees
from sac.utils import draw_trees


def main(*args):
    aligned = AlignedTrees(*args)
    # draw_trees(aligner.merged_src_tree, aligner.src_tree, include_word_idx=True)
    print(aligned.merged_src_tree)
    print(aligned.src_tree)
    print(aligned.ted)
    print(aligned.astred)
    print(aligned.n_null_aligns)
    print(aligned.label_changes)


if __name__ == "__main__":
    main("She asked me why I used to call her father Harold",
         "Ze vroeg waarom ik haar vader Harold noemde .",
         "0-0 1-1 3-2 4-3 5-7 6-7 7-7 8-4 9-5 10-6")
