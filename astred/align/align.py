from collections import defaultdict
from functools import cached_property
from itertools import combinations
from typing import (Dict, Generator, List, NamedTuple, Optional, Set, Tuple,
                    Union)

AlignedIdxs = NamedTuple("AlignedIdxs", [("src", int), ("tgt", int)])


class Align(list):
    def __init__(self, src, tgt, idxs: List[AlignedIdxs]):
        idxs = self._add_missing_aligns(src, tgt, idxs)

        super().__init__(idxs)

        self.src = src
        self.tgt = tgt

        self.src_idxs, self.tgt_idxs = self._get_src_tgt_idxs()
        self._create_directional_dict()
        self.align()
        self.set_cross()

    @cached_property
    def cross(self) -> int:
        src_cross = sum([item.cross for item in self.src])
        tgt_cross = sum([item.cross for item in self.tgt])

        assert src_cross == tgt_cross

        return src_cross

    def set_cross(self):
        for pairs in combinations(self, 2):
            # Do not align to "unaligned" indices
            if any(pair.src == -1 or pair.tgt == -1 for pair in pairs):
                continue

            pair1, pair2 = pairs

            # If: has cross
            if pair2.tgt < pair1.tgt:
                # Find the relative movement that each src_token has to do
                # We attribute the cross value to the word that moves furthest
                # If the distance is equal, the first pair gets the cross value
                # These values are already hidden in the Word's properties (not absolute)
                # but re-calculating based on the alignments is cheap and easier,
                # so let's do that.
                pair1_rel_mvmt = abs(pair1.src - pair1.tgt)
                pair2_rel_mvmt = abs(pair2.src - pair2.tgt)

                if pair2_rel_mvmt > pair1_rel_mvmt:
                    self.src[pair2.src].add_cross(pair2.tgt)
                    self.tgt[pair2.tgt].add_cross(pair2.src)
                else:
                    self.src[pair1.src].add_cross(pair1.tgt)
                    self.tgt[pair1.tgt].add_cross(pair1.src)

    @staticmethod
    def _add_missing_aligns(src, tgt, idxs):
        src_idxs, tgt_idxs = map(set, zip(*idxs))
        src_missing = [AlignedIdxs(idx, -1) for idx in range(len(src)) if idx not in src_idxs]
        tgt_missing = [AlignedIdxs(-1, idx) for idx in range(len(tgt)) if idx not in tgt_idxs]

        return sorted(idxs + src_missing + tgt_missing)

    def _get_src_tgt_idxs(self):
        """ Based on a list of alignments, get back the unique src and target indices """
        src_idxs, tgt_idxs = zip(*self)
        return sorted(set(src_idxs)), sorted(set(tgt_idxs))

    def _create_directional_dict(self):
        src_d_idxs = defaultdict(list)
        tgt_d_idxs = defaultdict(list)

        for pair in self:
            src_d_idxs[pair.src].append(pair.tgt)
            tgt_d_idxs[pair.tgt].append(pair.src)

        self.src2tgt_idxs = dict(src_d_idxs)
        self.tgt2src_idxs = dict(tgt_d_idxs)

    @staticmethod
    def alignments_cross(aligns: Tuple[AlignedIdxs]):
        """ Check if two alignments cross one another
            i.e., sort them and see if first item's tgt > second item's tgt"""
        aligns = sorted(aligns)
        a_1, a_2 = aligns

        return a_1.tgt > a_2.tgt

    def align(self):
        raise NotImplementedError
