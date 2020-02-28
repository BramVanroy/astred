from collections import defaultdict
from itertools import combinations


class Cross:
    def __init__(self, alignments):
        self.aligns = aligns_from_str(alignments)
        self.src_idxs, self.tgt_idxs = zip(*self.aligns)

        # Only do computationally heavy calculations when
        # the property is actually required
        self._aligns_w_null = None
        self._cross = None
        self._null_aligns = None
        self._seq_aligns = None
        self._seq_cross = None
        self._groups = None
        self._src2aligns_d = None
        self._tgt2aligns_d = None
        self._src2tgtlist_d = None
        self._tgt2srclist_d = None

    @property
    def aligns_w_null(self):
        if self._aligns_w_null is None:
            self._aligns_w_null = sorted(self.aligns + self.null_aligns)
        return self._aligns_w_null

    @property
    def src2aligns_d(self):
        if self._src2aligns_d is None:
            self._src2aligns_d, self._tgt2aligns_d = self._direction2aligns(self.aligns_w_null)
        return self._src2aligns_d

    @property
    def tgt2aligns_d(self):
        if self._tgt2aligns_d is None:
            self._src2aligns_d, self._tgt2aligns_d = self._direction2aligns(self.aligns_w_null)
        return self._tgt2aligns_d

    @property
    def cross(self):
        if self._cross is None:
            self._cross = self.get_cross(self.aligns)
        return self._cross

    @property
    def groups(self):
        if self._groups is None:
            self._groups = self._word_align_to_groups()
        return self._groups

    @property
    def null_aligns(self):
        if self._null_aligns is None:
            self._null_aligns = self._get_null_aligns()
        return self._null_aligns

    @property
    def seq_aligns(self):
        if self._seq_aligns is None:
            self._seq_aligns = self._groups_to_seq_align()
        return self._seq_aligns

    @property
    def seq_cross(self):
        if self._seq_cross is None:
            self._seq_cross = self.get_cross(self.seq_aligns)
        return self._seq_cross

    @property
    def src2tgtlist_d(self):
        if self._src2tgtlist_d is None:
            self._src2tgtlist_d = {src: [i[1] for i in align] for src, align in self.src2aligns_d.items()}
        return self._src2tgtlist_d

    @property
    def tgt2srclist_d(self):
        if self._tgt2srclist_d is None:
            self._tgt2srclist_d = {tgt: [i[0] for i in align] for tgt, align in self.tgt2aligns_d.items()}
        return self._tgt2srclist_d

    @classmethod
    def from_list(cls, align_list, *args, **kwargs):
        return cls(aligns_to_str(align_list))

    def _word_align_to_groups(self):
        """ Get all possible combinations of src_idxs and tgt_idxs (min_size=2), and find groups between these src/tgt
            combos so that the following requirements are met for that group:
            1. no item in src or tgt can be aligned to anything outside the group
            2. no alignments within the group can cross each other

            After finding all possible combinations, it is likely that not all items have been grouped. In that case,
            check which alignments are not grouped yet, and add them to their individual group.

            Variable names:
            - (src|tgt)2aligns_d: {idx: list of alignments (as tuples) with this idx}
            - (src2tgt|tgt2src)list_d: {src_idx|tgt_idx: list of tgts or srcs (resp.) connected to this item}
            - (src|tgt)_combs: all possible combinations of src or tgt idxs (with a min_size=2) and excluding combos
                               that are aligned to -1 (= null alignments; they break groups)
            - (src|tgt)_idxs: indices of src or tgt, not including null alignments
            - (src|tgt)_idxs_grouped: sets to keep track which idxs have already been put into a group
        """
        src_combs = self._consec_combinations(list(self.src2tgtlist_d.keys()), self.src2tgtlist_d)
        tgt_combs = self._consec_combinations(list(self.tgt2srclist_d.keys()), self.tgt2srclist_d)

        src_idxs_grouped = set()
        tgt_idxs_grouped = set()
        groups = []
        # Try grouping a src_comb with a tgt_comb
        for src_comb in src_combs:
            # If any item in this combination has already been grouped in another combo group, continue
            if any(src in src_idxs_grouped for src in src_comb):
                continue

            for tgt_comb in tgt_combs:
                # If any item in this combination has already been grouped in another combo group, continue
                if any(tgt in tgt_idxs_grouped for tgt in tgt_comb):
                    continue
                has_external_aligns = self._has_external_aligns(
                    src_comb, tgt_comb, self.src2tgtlist_d, self.tgt2srclist_d
                )

                # If the src_combo+tgt_combo have no external_aligns, keep going
                if not has_external_aligns:
                    has_internal_cross = self._has_internal_cross(src_comb, self.src2aligns_d)
                    # If the src_combo+tgt_combo have no internal_crosses, they can form a group
                    if not has_internal_cross:
                        # Keep track of src+tgt idxs that are already grouped
                        src_idxs_grouped.update(src_comb)
                        tgt_idxs_grouped.update(tgt_comb)
                        # Get all alignments of this group and add them as group
                        alignments_of_group = [i for src in src_comb for i in self.src2aligns_d[src]]
                        groups.append(alignments_of_group)
                        # Break because we have found a suitable group
                        break

        # Manually checking if all alignments are grouped, and if not: adding as their own group, solves that
        for src_id in self.src_idxs:
            if src_id not in src_idxs_grouped:
                for aligns in self.src2aligns_d[src_id]:
                    if [aligns] not in groups:
                        groups.append([aligns])

        for tgt_id in self.tgt_idxs:
            if tgt_id not in tgt_idxs_grouped:
                for aligns in self.tgt2aligns_d[tgt_id]:
                    if [aligns] not in groups:
                        groups.append([aligns])

        return sorted(groups)

    def _groups_to_seq_align(self):
        """ Takes groups of sequential alignments and transforms it into
            actual sequence alignments with one alignment point per sequence. """
        max_src, max_tgt = [], []
        # group is a consecutive group of alignments
        for group in self.groups:
            """ Get a group's max source/target """
            src, tgt = map(list, zip(*group))
            max_src.append(max(src))
            max_tgt.append(max(tgt))

        # Get the indices of the values in ascending order
        src_order = self._get_idxs(max_src)
        tgt_order = self._get_idxs(max_tgt)

        # Merge indices into sequence alignments
        seq_align = [*zip(src_order, tgt_order)]

        return seq_align

    @staticmethod
    def _get_idxs(l):
        """ Convert values to indices. This ensure that there are no strange gaps
            between sequence alignments (e.g. when an index is not word-aligned) """
        l_sort = sorted(list(set(l)))
        l_idxs = [l_sort.index(x) for x in l]
        return l_idxs

    @staticmethod
    def _get_src_tgt_idxs(aligns):
        """ Based on a list of alignments, get back the unique src and target indices """
        src_idxs, tgt_idxs = zip(*aligns)
        src_idxs = sorted(set(src_idxs))
        tgt_idxs = sorted(set(tgt_idxs))

        return src_idxs, tgt_idxs

    def _has_internal_cross(self, src_comb, src2aligns_d):
        """ Check if alignments in this src_comb cross each other.
            No need to check tgt_comb, as 'cross' is direction-agnostic """
        # Get alignments of src_comb indices
        aligns = []
        for idx in src_comb:
            aligns.extend(src2aligns_d[idx])

        # Check if any alignment combination crosses each other
        return any(self.alignments_cross(align_pair) for align_pair in combinations(aligns, 2))

    @staticmethod
    def alignments_cross(aligns):
        """ Check if two alignments cross one another
            i.e., sort them and see if first item's tgt > second item's tgt"""
        aligns = sorted(aligns)
        a_1, a_2 = aligns

        return a_1[1] > a_2[1]

    @staticmethod
    def _has_external_aligns(src_comb, tgt_comb, src2tgtlist_d, tgt2srclist_d):
        """ If any source or target id is connected to an item that is not in the other combo,
            i.e. that is outside this group, return True. """
        for src_idx in src_comb:
            if any(i not in tgt_comb for i in src2tgtlist_d[src_idx]):
                return True

        for tgt_idx in tgt_comb:
            if any(i not in src_comb for i in tgt2srclist_d[tgt_idx]):
                return True

        return False

    @staticmethod
    def _direction2aligns(tuple_aligns):
        """ Get dictionaries {src_idx|tgt_idx: list of alignments (as tuples) with this idx}"""
        d_src = defaultdict(list)
        d_tgt = defaultdict(list)

        for align in tuple_aligns:
            d_src[align[0]].append(align)
            d_tgt[align[1]].append(align)

        return dict(d_src), dict(d_tgt)

    def _get_null_aligns(self):
        """ Get missing idxs (= null alignments) and return them as alignments to -1.
            We use -1 so that we can still order our lists containing null alignments.
            Expects SORTED input lists.
        """

        def missing_idxs(idxs):
            """ Note that this _only_ works on the alignments. This implies
                that when words at the end of a sequence are not aligned
                (so their index is missing), this will NOT be caught by this method.
                To catch that, we need to know the number of tokens.
                We can override this method in SAC to do that. """
            idxs = set(idxs)
            max_idx = max(idxs)
            return [i for i in range(max_idx) if i not in idxs]

        src_missing = [(idx, -1) for idx in missing_idxs(self.src_idxs)]
        tgt_missing = [(-1, idx) for idx in missing_idxs(self.tgt_idxs)]

        return src_missing + tgt_missing

    @staticmethod
    def _consec_combinations(idxs, dir2dirlist_d=None):
        """ Get all consequtive combinations of idxs of all possible lengths.
            When getting consecutive combinations in cross, we want to split on -1 (null),
            but when getting consec groups in SAC, we already have groups without -1, so
            no need to check (dir2dirlist_d will be None).
            Sort by largest so that we can find the largest possible groups first"""
        idxs.sort()
        c = []
        for i in range(len(idxs)):
            for j in range(i + 1, len(idxs) + 1):
                s = idxs[i:j]
                # Do not make combinations with -1 (null), because -1 should always break groups
                if dir2dirlist_d is not None and (-1 in s or any(-1 in dir2dirlist_d[i] for i in s)):
                    continue

                c.append(s)

        return sorted(c, key=len, reverse=True)

    @staticmethod
    def get_cross(aligns):
        """ Get all crosses of a given list of alignments.

        :param aligns: a str representing alignments
        :return: the cross value for these alignments
        """

        tgt_idxs = [pair[1] for pair in aligns]

        return sum([1 for t1, t2 in combinations(tgt_idxs, 2) if t2 < t1])


def aligns_from_str(aligns):
    return sorted([tuple(map(int, align.split("-"))) for align in aligns.split()])


def aligns_to_str(aligns):
    """ Convert list of alignments (tuple of src, tgt) to GIZA/Pharaoh string """
    return " ".join([f"{s}-{t}" for s, t in aligns])
