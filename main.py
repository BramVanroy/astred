from sac import Cross, SAC
from nltk.draw import draw_trees

def main(*args):
    sac = SAC(*args)

    print("WORD ALIGN", sac.aligns_str)
    print("WORD CROSS", sac.cross)

    print("SEQ ALIGN", sac.seq_aligns_str)
    print("SEQ CROSS", sac.seq_cross)
    print(sac.groups)
    src_tree = sac.src_tree

    sac.grouped_trees()
    draw_trees(src_tree)


if __name__ == "__main__":
    main("0-0 1-2 2-1 4-3 5-4 6-8 7-8 8-8 9-5 10-6 11-7 12-9",
         "Sometimes she asks me why I used to call her father Harold .",
         "Soms vraagt ze waarom ik haar vader Harold noemde .")
