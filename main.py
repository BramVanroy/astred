from sac import ASTrED
from sac.utils import draw_trees


def main(*args):
    aligned = ASTrED(*args)
    aligned_mwe = ASTrED(*args, group_mwe=True)
    # draw_trees(aligner.merged_src_tree, aligner.src_tree, include_word_idx=True)
    print(aligned.seq_cross)
    print(aligned_mwe.seq_cross)
    print(aligned_mwe.sac_cross)
    print(aligned_mwe.mwe_groups)
    print(aligned_mwe.sac_groups)
    print(aligned_mwe.mwe_src_idxs)


if __name__ == "__main__":
    # main("It is thought that there are 10,000 such deaths a year .",
    #      "Er zijn naar schatting 10.000 zulke sterfgevallen per jaar .",
    #      "0-2 0-3 1-2 1-3 2-2 2-3 3-2 3-3 4-0 5-1 6-4 7-5 8-6 9-7 10-8 11-9")

    main("Yesterday I set the alarm clock",
         "Ik zette gisteren de alarmklok",
         "0-2 1-0 2-1 3-3 4-4 5-4")

    # MWE should also include one-to-many and many-to-one
    # when in seq a large group is created (because largest combos are tested first)
    # this might still be split up in smaller groups by SAC. Find a way to ensure that
    # MWE in that larger group are not split up, even though they are not found then yet
