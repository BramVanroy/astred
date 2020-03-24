from collections import defaultdict
from itertools import combinations
from typing import Dict, Generator, List, Optional, Set, Tuple, Union

from .utils import AlignmentPair, aligns_from_str, aligns_to_str


class _Cross:
    def __init__(self, alignments: str, group_mwe: bool = False):
        self.aligns = aligns_from_str(alignments)
        self.src_idxs, self.tgt_idxs = self._get_src_tgt_idxs(self.aligns)
        self.group_mwe = group_mwe

        # Only do computationally heavy calculations when
        # the property is actually required
        self._aligns_w_null = None
        self._cross = None
        self._null_aligns = None

        self._seq_aligns = None
        self._seq_cross = None
        self._seq_groups = None

        self._src2aligns_d = None
        self._tgt2aligns_d = None
        self._src2tgtlist_d = None
        self._tgt2srclist_d = None

        self._mwe_groups = None
        self._mwe_src_idxs = None
        self._mwe_tgt_idxs = None

    @property
    def aligns_w_null(self):
        if self._aligns_w_null is None:
            self._aligns_w_null = sorted(self.aligns + self.null_aligns)
        return self._aligns_w_null

    @property
    def cross(self):
        if self._cross is None:
            self._cross = self._get_cross(self.aligns)
        return self._cross

    @property
    def mwe_groups(self):
        if self._mwe_groups is None:
            self._mwe_groups = self._find_mwe()
        return self._mwe_groups

    @property
    def mwe_src_idxs(self):
        if self._mwe_src_idxs is None:
            self._mwe_src_idxs, self._mwe_tgt_idxs = self._get_mwe_idxs()
        return self._mwe_src_idxs

    @property
    def mwe_tgt_idxs(self):
        if self._mwe_tgt_idxs is None:
            self._mwe_src_idxs, self._mwe_tgt_idxs = self._get_mwe_idxs()
        return self._mwe_tgt_idxs

    @property
    def null_aligns(self):
        if self._null_aligns is None:
            self._null_aligns = self._get_null_aligns()
        return self._null_aligns

    @property
    def seq_aligns(self):
        if self._seq_aligns is None:
            self._seq_aligns = self._groups_to_align(self.seq_groups)
        return self._seq_aligns

    @property
    def seq_cross(self):
        if self._seq_cross is None:
            self._seq_cross = self._get_cross(self.seq_aligns)
        return self._seq_cross

    @property
    def seq_groups(self):
        if self._seq_groups is None:
            self._seq_groups = self._word_align_to_groups()
        return self._seq_groups

    @property
    def src2aligns_d(self):
        if self._src2aligns_d is None:
            self._src2aligns_d, self._tgt2aligns_d = self._direction2aligns(self.aligns_w_null)
        return self._src2aligns_d

    @property
    def src2tgtlist_d(self):
        if self._src2tgtlist_d is None:
            self._src2tgtlist_d = {
                src: [i.tgt for i in align] for src, align in self.src2aligns_d.items()
            }
        return self._src2tgtlist_d

    @property
    def tgt2aligns_d(self):
        if self._tgt2aligns_d is None:
            self._src2aligns_d, self._tgt2aligns_d = self._direction2aligns(self.aligns_w_null)
        return self._tgt2aligns_d

    @property
    def tgt2srclist_d(self):
        if self._tgt2srclist_d is None:
            self._tgt2srclist_d = {
                tgt: [i.src for i in align] for tgt, align in self.tgt2aligns_d.items()
            }
        return self._tgt2srclist_d

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
                for aligns in self.src2aligns_d[src_idx]:
                    if [aligns] not in groups:
                        groups.append([AlignmentPair(*aligns)])

        for tgt_idx in self.tgt_idxs:
            if tgt_idx not in tgt_idxs_grouped:
                for aligns in self.tgt2aligns_d[tgt_idx]:
                    if [aligns] not in groups:
                        groups.append([AlignmentPair(*aligns)])

        return groups

    def _find_mwe(self):
        # MWE can only exist between src and target that both have more than one element
        # That means that we do not catch one-to-many, many-to-one because that would be
        # too broad and have too many false positives
        src_combs = self._consec_combinations(
            list(self.src2tgtlist_d.keys()), self.src2tgtlist_d, min_size=2
        )

        # need to listify the generator because it is in the internal loop
        # the generator would exhaust after the first outer loop, but we need to re-use
        # it for all outer loops, so build a list
        tgt_combs = list(
            self._consec_combinations(
                list(self.tgt2srclist_d.keys()), self.tgt2srclist_d, min_size=2
            )
        )

        src_idxs_grouped = set()
        tgt_idxs_grouped = set()
        mwe_groups = []

        # Try grouping a src_comb with a tgt_comb
        for src_comb in src_combs:
            # If any item in this combination has already been grouped in another
            # combo group, continue
            if any(src in src_idxs_grouped for src in src_comb):
                continue

            for tgt_comb in tgt_combs:
                # If any item in this combination has already been grouped in another
                # combo group, continue
                if any(tgt in tgt_idxs_grouped for tgt in tgt_comb):
                    continue

                if self._is_mwe(src_comb, tgt_comb):
                    # Keep track of src+tgt idxs that are already grouped
                    src_idxs_grouped.update(src_comb)
                    tgt_idxs_grouped.update(tgt_comb)

                    mwe_groups.append((src_comb, tgt_comb))
                    # Break because we have found a suitable group
                    break

        return mwe_groups

    def _get_mwe_idxs(self):
        src, tgt = set(), set()
        for src_idxs, tgt_idxs in self.mwe_groups:
            src.update(src_idxs)
            tgt.update(tgt_idxs)

        return src, tgt

    def _get_null_aligns(self):
        """ Get missing idxs (= null alignments) and return them as alignments to -1.
            We use -1 so that we can still order our lists containing null alignments.
            Expects SORTED input lists.
        """

        def missing_idxs(idxs: List):
            """ Note that this _only_ works on the alignments. This implies
                that when words at the end of a sequence are not aligned
                (so their index is missing), this will NOT be caught by this method.
                To catch that, we need to know the number of tokens.
                We can override this method in SACr to do that. """
            idxs = set(idxs)
            max_idx = max(idxs)
            return [i for i in range(max_idx) if i not in idxs]

        src_missing = [AlignmentPair(idx, -1) for idx in missing_idxs(self.src_idxs)]
        tgt_missing = [AlignmentPair(-1, idx) for idx in missing_idxs(self.tgt_idxs)]

        return src_missing + tgt_missing

    def _groups_to_align(self, groups: List):
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
        aligns = [AlignmentPair(src, tgt) for src, tgt in zip(src_order, tgt_order)]

        return aligns

    def _has_external_aligns(self, src_comb: Set[int], tgt_comb: Set[int]):
        """ If any source or target id is connected to an item that is not in the other combo,
            i.e. that is outside this group, return True. """
        for src_idx in src_comb:
            if any(i not in tgt_comb for i in self.src2tgtlist_d[src_idx]):
                return True

        for tgt_idx in tgt_comb:
            if any(i not in src_comb for i in self.tgt2srclist_d[tgt_idx]):
                return True

        return False

    def _has_internal_cross(self, src_comb: List):
        """ Check if alignments in this src_comb cross each other.
            No need to check tgt_comb, as 'cross' is direction-agnostic """
        # Get alignments of src_comb indices
        aligns = []
        for idx in src_comb:
            aligns.extend(self.src2aligns_d[idx])

        # Check if any alignment combination crosses each other
        return any(self.alignments_cross(align_pair) for align_pair in combinations(aligns, 2))

    def _is_in_mwe(self, src_comb: List, tgt_comb: List):
        for src_idxs, tgt_idxs in self.mwe_groups:
            if src_idxs == src_comb and tgt_idxs == tgt_comb:
                return True
        return False

    def _is_mwe(self, src_comb: Set[int], tgt_comb: Set[int]):
        """ If all source items are connected to all target items, we assume that this is
            a segment that is likely to be a multi-word expression.
            We do NOT count one-to-many or many-to-one alignments as MWE because
            that would be too broad.
            """
        for src_idx in src_comb:
            if any(
                tgt_idx not in self._src2tgtlist_d[src_idx] for tgt_idx in tgt_comb
            ) or self._has_external_aligns(src_comb, tgt_comb):
                return False

        return True

    def _word_align_to_groups(self):
        """ Get all possible combinations of src_idxs and tgt_idxs (min_size=2), and find
            groups between these src/tgt combos so that the following requirements are met for
            that group:
            1. no item in src or tgt can be aligned to anything outside the group
            2. no alignments within the group can cross each other

            After finding all possible combinations, it is likely that not all items have been
            grouped. In that case, check which alignments are not grouped yet, and add them to
            their individual group.

            Variable names:
            - (src|tgt)2aligns_d: {idx: list of alignments (as tuples) with this idx}
            - (src2tgt|tgt2src)list_d: {src_idx|tgt_idx: list of tgts or srcs (resp.) connected
            to this item}
            - (src|tgt)_combs: all possible combinations of src or tgt idxs (with a min_size=2)
             and excluding combos
                               that are aligned to -1 (= null alignments; they break groups)
            - (src|tgt)_idxs: indices of src or tgt, not including null alignments
            - (src|tgt)_idxs_grouped: sets to keep track which idxs have already been put into
             a group
        """
        src_combs = self._consec_combinations(
            list(self.src2tgtlist_d.keys()), self.src2tgtlist_d
        )

        # need to listify the generator because it is in the internal loop
        # the generator would exhaust after the first outer loop, but we need to re-use
        # it for all outer loops, so build a list
        tgt_combs = list(
            self._consec_combinations(list(self.tgt2srclist_d.keys()), self.tgt2srclist_d)
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
            n_src_items = len(src_comb)
            # only calculate internal_cross once for src_comb
            has_internal_cross = self._has_internal_cross(src_comb)

            for tgt_comb in tgt_combs:
                # If any item in this combination has already been grouped in another
                # combo group, continue
                if any(tgt in tgt_idxs_grouped for tgt in tgt_comb):
                    continue

                n_tgt_items = len(tgt_comb)
                is_mwe = False
                if n_src_items > 1 and n_tgt_items > 1:
                    is_mwe = self.group_mwe and self._is_in_mwe(src_comb, tgt_comb)

                # If the src_combo+tgt_combo is a MWE or (it has no external aligns and
                # no internal crosses): go on
                if is_mwe or (
                    not self._has_external_aligns(src_comb, tgt_comb) and not has_internal_cross
                ):
                    # Keep track of src+tgt idxs that are already grouped
                    src_idxs_grouped.update(src_comb)
                    tgt_idxs_grouped.update(tgt_comb)
                    # Get all alignments of this group and add them as group
                    alignments_of_group = sorted(
                        [i for src in src_comb for i in self.src2aligns_d[src]]
                    )

                    groups.append(alignments_of_group)
                    # Break because we have found a suitable group
                    break

        groups = self._add_unsolved_idxs(src_idxs_grouped, tgt_idxs_grouped, groups)

        return sorted(groups)

    @staticmethod
    def alignments_cross(aligns: Tuple[AlignmentPair]):
        """ Check if two alignments cross one another
            i.e., sort them and see if first item's tgt > second item's tgt"""
        aligns = sorted(aligns)
        a_1, a_2 = aligns

        return a_1.tgt > a_2.tgt

    @staticmethod
    def _consec_combinations(
        idxs: List[int], dir2dirlist_d: Optional[Dict] = None, min_size: int = 1
    ) -> Generator:
        """ Get all consequtive combinations of idxs of all possible lengths.
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

    @staticmethod
    def _direction2aligns(
        tuple_aligns: List[AlignmentPair],
    ) -> Tuple[Dict[int, List[AlignmentPair]], Dict[int, List[AlignmentPair]]]:
        """ Get dictionaries {src_idx|tgt_idx: list of alignments (as tuples) with this idx}"""
        d_src = defaultdict(list)
        d_tgt = defaultdict(list)

        for align in tuple_aligns:
            d_src[align.src].append(align)
            d_tgt[align.tgt].append(align)

        return dict(d_src), dict(d_tgt)

    @staticmethod
    def _get_idxs(idxs: List[int]):
        """ Convert values to indices. This ensure that there are no strange gaps
            between sequence alignments (e.g. when an index is not word-aligned) """
        l_sort = sorted(list(set(idxs)))

        return [l_sort.index(x) for x in idxs]

    @staticmethod
    def _get_cross(aligns: List[AlignmentPair]):
        """ Get all crosses of a given list of alignments.

        :param aligns: a str representing alignments
        :return: the cross value for these alignments
        """
        tgt_idxs = [pair.tgt for pair in aligns]

        return sum([1 for t1, t2 in combinations(tgt_idxs, 2) if t2 < t1])

    @staticmethod
    def _get_src_tgt_idxs(aligns: List[Union[Tuple[int, int], AlignmentPair]]):
        """ Based on a list of alignments, get back the unique src and target indices """
        src_idxs, tgt_idxs = zip(*aligns)
        return sorted(set(src_idxs)), sorted(set(tgt_idxs))

    @classmethod
    def from_list(
        cls, align_list: List[Union[Tuple[int, int], AlignmentPair]], *args, **kwargs
    ):
        return cls(aligns_to_str(align_list), *args, **kwargs)
