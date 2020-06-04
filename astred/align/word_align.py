from collections import defaultdict
from functools import cached_property
from itertools import combinations
from typing import Dict, Generator, List, Optional, Set

from .align import Align, AlignedIdxs


class WordAlign(Align):
    def __init__(self, src, tgt, idxs: List[AlignedIdxs]):
        super().__init__(src, tgt, idxs)

    def _create_directional_dict(self):
        # Overwrite from base class because we want indexes but als src2tgt
        # src2tgt is not possible in sequences because there, items can be lists
        # and lists cannot be used as dictionary keys (not hashable)
        src_d = defaultdict(list)
        tgt_d = defaultdict(list)

        src_d_idxs = defaultdict(list)
        tgt_d_idxs = defaultdict(list)

        for pair in self:
            src_d_idxs[pair.src].append(pair.tgt)
            tgt_d_idxs[pair.tgt].append(pair.src)

            # Do not create dict with null alignments (because -1 will select last item)
            if pair.src > -1 and pair.tgt > -1:
                src_d[self.src[pair.src]].append(self.tgt[pair.tgt])
                src_d[self.tgt[pair.tgt]].append(self.src[pair.src])

        self.src2tgt = dict(src_d)
        self.tgt2src = dict(tgt_d)

        self.src2tgt_idxs = dict(src_d_idxs)
        self.tgt2src_idxs = dict(tgt_d_idxs)

    @cached_property
    def cross(self) -> int:
        src_word_cross = self.src.word_cross
        tgt_word_cross = self.tgt.word_cross

        assert src_word_cross == tgt_word_cross

        return src_word_cross

    #########################################
    # NAIVELY GROUPING WORDS INTO SEQUENCES #
    #########################################
    def word_align_to_groups(self):
        """Get all possible combinations of src_idxs and tgt_idxs (min_size=2), and find
           groups between these src/tgt combos so that the following requirements are met for
           that group:
             1. no item in src or tgt can be aligned to anything outside the group
             2. no alignments within the group can cross each other

           After finding all possible combinations, it is likely that not all items have been
           grouped. In that case, check which alignments are not grouped yet, and add them to
           their individual group.
        """
        src_combs = self._consec_combinations(list(self.src2tgt_idxs.keys()), self.src2tgt_idxs)

        # need to listify the generator because it is in the internal loop
        # the generator would exhaust after the first outer loop, but we need to re-use
        # it for all outer loops, so build a list
        tgt_combs = list(
            self._consec_combinations(list(self.tgt2src_idxs.keys()), self.tgt2src_idxs)
        )

        src_idxs_grouped = set()
        tgt_idxs_grouped = set()
        groups = []

        # Try grouping a src_comb with a tgt_comb
        for src_comb in src_combs:
            # If any item in this combination has already been grouped in another
            # combo group, continue
            if any(src in src_idxs_grouped for src in src_comb):
                continue

            # only calculate internal_cross once for src_comb
            has_internal_cross = self._has_internal_cross(src_comb)

            for tgt_comb in tgt_combs:
                # If any item in this combination has already been grouped in another
                # combo group, continue
                if any(tgt in tgt_idxs_grouped for tgt in tgt_comb):
                    continue

                # If the src_combo+tgt_combo has no external aligns and no internal crosses: go on
                if not self._has_external_aligns(src_comb, tgt_comb) and not has_internal_cross:
                    # Keep track of src+tgt idxs that are already grouped
                    src_idxs_grouped.update(src_comb)
                    tgt_idxs_grouped.update(tgt_comb)
                    # Get all alignments of this group and add them as group
                    alignments_of_group = sorted(
                        [
                            AlignedIdxs(src, i)
                            for src in src_comb
                            for i in self.src2tgt_idxs[src]
                        ]
                    )

                    groups.append(alignments_of_group)
                    # Break because we have found a suitable group
                    break

        groups = self._add_unsolved_idxs(src_idxs_grouped, tgt_idxs_grouped, groups)

        return sorted(groups)

    def _add_unsolved_idxs(
        self, src_idxs_grouped: Set[int], tgt_idxs_grouped: Set[int], groups: List
    ):
        """ Manually checking if all alignments are grouped, and if not: adding as
        their own group, solves that
        :param src_idxs_grouped:
        :param tgt_idxs_grouped:
        :param groups:
        :return:
        """
        for src_idx in self.src_idxs:
            if src_idx not in src_idxs_grouped:
                for tgt_idx in self.src2tgt_idxs[src_idx]:
                    aligns = [AlignedIdxs(src_idx, tgt_idx)]
                    if aligns not in groups:
                        groups.append(aligns)

        for tgt_idx in self.tgt_idxs:
            if tgt_idx not in tgt_idxs_grouped:
                for src_idx in self.tgt2src_idxs[tgt_idx]:
                    aligns = [AlignedIdxs(src_idx, tgt_idx)]
                    if aligns not in groups:
                        groups.append(aligns)

        return groups

    def groups_to_align(self, groups: List):
        """ Takes groups of alignments and transforms it into
            actual alignments with one alignment point per group. """
        max_src, max_tgt = [], []
        # group is a consecutive group of alignments
        for group in groups:
            """ Get a group's max source/target """
            src, tgt = map(list, zip(*group))
            max_src.append(max(src))
            max_tgt.append(max(tgt))

        # Get the indices of the values in ascending order
        src_order = self._get_idxs(max_src)
        tgt_order = self._get_idxs(max_tgt)

        # Merge indices into sequence alignments
        aligns = [AlignedIdxs(src, tgt) for src, tgt in zip(src_order, tgt_order)]

        return aligns

    @staticmethod
    def _get_idxs(idxs: List[int]):
        """ Convert values to indices. This ensure that there are no strange gaps
            between sequence alignments (e.g. when an index is not word-aligned) """
        l_sort = sorted(list(set(idxs)))

        return [l_sort.index(x) for x in idxs]

    @staticmethod
    def _consec_combinations(
        idxs: List[int], dir2dirlist_d: Optional[Dict] = None, min_size: int = 1
    ) -> Generator:
        """Get all consequtive combinations of idxs of all possible lengths.
            When getting consecutive combinations in cross, we want to split on -1 (null),
            but when getting consec groups in SAC, we already have groups without -1, so
            no need to check (dir2dirlist_d will be None).
            Returns largest possible groups first and decreases in length """
        idxs.sort()
        n_idxs = len(idxs)
        for i in range(n_idxs, min_size - 1, -1):
            for j in range(n_idxs - i + 1):
                s = set(idxs[j : j + i])
                # Do not make combinations with -1 (null), because -1 should always break groups
                if dir2dirlist_d is not None and (
                    -1 in s or any(-1 in dir2dirlist_d[i] for i in s)
                ):
                    continue
                yield s

    def _has_internal_cross(self, src_comb: List):
        """ Check if alignments in this src_comb cross each other.
            No need to check tgt_comb, as 'cross' is direction-agnostic """
        # Get alignments of src_comb indices
        aligns = []
        for idx in src_comb:
            for tgt_idx in self.src2tgt_idxs[idx]:
                aligns.append(AlignedIdxs(idx, tgt_idx))

        # Check if any alignment combination crosses each other
        return any(self.alignments_cross(align_pair) for align_pair in combinations(aligns, 2))

    def _has_external_aligns(self, src_comb: Set[int], tgt_comb: Set[int]):
        """ If any source or target id is connected to an item that is not in the other combo,
            i.e. that is outside this group, return True. """
        for src_idx in src_comb:
            if any(i not in tgt_comb for i in self.src2tgt_idxs[src_idx]):
                return True

        for tgt_idx in tgt_comb:
            if any(i not in src_comb for i in self.tgt2src_idxs[tgt_idx]):
                return True

        return False

    def align(self):
        # Add properties to words so they know who they are aligned with
        self.src.aligned_with = self.tgt
        self.tgt.aligned_with = self.src

        for src_idx, tgt_idx in self:
            # Do not align to "unaligned" indices
            if src_idx == -1 or tgt_idx == -1:
                continue

            self.src[src_idx].add_aligned_with(self.tgt[tgt_idx])
            self.tgt[tgt_idx].add_aligned_with(self.src[src_idx])
