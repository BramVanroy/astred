from sac import Cross


def main(aligns):
    cross = Cross.from_list([(0, 0), (1, 1), (2, 0), (3, 2), (4, 3), (4, 4), (5, 5)])
    print('WORD ALIGN', cross.aligns_str)
    print('WORD CROSS', cross.cross)

    print('SEQ ALIGN', cross.seq_aligns_str)
    print('SEQ CROSS', cross.seq_cross)


if __name__ == '__main__':
    alignments = '0-0 1-1 2-0 3-2 4-3 4-4 5-5'
    main(alignments)
