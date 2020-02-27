from sac import Cross, SAC, aligns_to_str
from nltk.draw import draw_trees

def main(*args):
    sac = SAC(*args)

    print()
    print("SAC WORD ALIGN", aligns_to_str(sac.aligns))
    print("SAC GROUPS", sac.groups)
    print("SAC SEQ ALIGN", aligns_to_str(sac.seq_aligns))
    sac.regroup_by_subtrees()


if __name__ == "__main__":
    main("0-0 1-1 3-2 4-3 5-7 6-7 7-7 8-4 9-5 10-6",
         "She asked me why I used to call her father Harold",
         "Ze vroeg waarom ik haar vader Harold noemde .")
