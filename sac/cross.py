from collections import defaultdict
from itertools import combinations


class Cross:
    def __init__(self, alignments, allow_mwe=False):
        self.aligns = aligns_from_str(alignments)
        self.src_idxs, self.tgt_idxs = zip(*self.aligns)
        self.allow_mwe = allow_mwe

        if allow_mwe:
            raise NotImplementedError("Even though a cool idea, this is not implemented yet.")

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
        self._mwe_src_group_to_tgt_group = None

    @property
    def mwe_src_group_to_tgt_group(self):
        raise NotImplementedError()

    @property
    def aligns_w_null(self):
        if self._aligns_w_null is None:
            self._aligns_w_null = sorted(self.aligns + self.null_aligns)
        return self._aligns_w_null

    @property
    def mwe_groups(self):
        return self._mwe_groups

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
    def seq_groups(self):
        if self._seq_groups is None:
            self._seq_groups = self._word_align_to_groups()
        return self._seq_groups

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
            self._seq_cross = self.get_cross(self.seq_aligns)
        return self._seq_cross

    @property
    def src2tgtlist_d(self):
        if self._src2tgtlist_d is None:
            self._src2tgtlist_d = {
                src: [i[1] for i in align] for src, align in self.src2aligns_d.items()
            }
        return self._src2tgtlist_d

    @property
    def tgt2srclist_d(self):
        if self._tgt2srclist_d is None:
            self._tgt2srclist_d = {
                tgt: [i[0] for i in align] for tgt, align in self.tgt2aligns_d.items()
            }
        return self._tgt2srclist_d

    @classmethod
    def from_list(cls, align_list, *args, **kwargs):
        return cls(aligns_to_str(align_list))

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
        tgt_combs = self._consec_combinations(
            list(self.tgt2srclist_d.keys()), self.tgt2srclist_d
        )

        src_idxs_grouped = set()
        tgt_idxs_grouped = set()
        groups = []
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
                has_external_aligns = self._has_external_aligns(
                    src_comb, tgt_comb
                )

                # If the src_combo+tgt_combo have no external_aligns, keep going
                if not has_external_aligns:
                    has_internal_cross = self._has_internal_cross(src_comb)
                    # only execute _is_mwe if it is allowed
                    is_mwe = self.allow_mwe and self._is_mwe(src_comb, tgt_comb)
                    # If the src_combo+tgt_combo have no internal_crosses, they can form a group
                    if not has_internal_cross or is_mwe:
                        # Keep track of src+tgt idxs that are already grouped
                        src_idxs_grouped.update(src_comb)
                        tgt_idxs_grouped.update(tgt_comb)
                        # Get all alignments of this group and add them as group
                        alignments_of_group = [
                            i for src in src_comb for i in self.src2aligns_d[src]
                        ]
                        groups.append(alignments_of_group)
                        if is_mwe:
                            mwe_groups.append(alignments_of_group)
                        # Break because we have found a suitable group
                        break

        groups = self._add_unsolved_idxs(src_idxs_grouped, tgt_idxs_grouped, groups)
        self._mwe_groups = mwe_groups

        return sorted(groups)

    def _add_unsolved_idxs(self, src_idxs_grouped, tgt_idxs_grouped, groups):
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
                        groups.append([aligns])

        for tgt_idx in self.tgt_idxs:
            if tgt_idx not in tgt_idxs_grouped:
                for aligns in self.tgt2aligns_d[tgt_idx]:
                    if [aligns] not in groups:
                        groups.append([aligns])

        return groups

    def _groups_to_align(self, groups):
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
        aligns = [*zip(src_order, tgt_order)]

        return aligns

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

    def _has_internal_cross(self, src_comb):
        """ Check if alignments in this src_comb cross each other.
            No need to check tgt_comb, as 'cross' is direction-agnostic """
        # Get alignments of src_comb indices
        aligns = []
        for idx in src_comb:
            aligns.extend(self.src2aligns_d[idx])

        # Check if any alignment combination crosses each other
        return any(self.alignments_cross(align_pair) for align_pair in combinations(aligns, 2))

    @staticmethod
    def alignments_cross(aligns):
        """ Check if two alignments cross one another
            i.e., sort them and see if first item's tgt > second item's tgt"""
        aligns = sorted(aligns)
        a_1, a_2 = aligns

        return a_1[1] > a_2[1]

    def _has_external_aligns(self, src_comb, tgt_comb):
        """ If any source or target id is connected to an item that is not in the other combo,
            i.e. that is outside this group, return True. """
        for src_idx in src_comb:
            if any(i not in tgt_comb for i in self.src2tgtlist_d[src_idx]):
                return True

        for tgt_idx in tgt_comb:
            if any(i not in src_comb for i in self.tgt2srclist_d[tgt_idx]):
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
        n_idxs = len(idxs)
        for i in range(n_idxs, 0, -1):
            for j in range(n_idxs - i + 1):
                s = idxs[j:j + i]
                # Do not make combinations with -1 (null), because -1 should always break groups
                if dir2dirlist_d is not None and (
                    -1 in s or any(-1 in dir2dirlist_d[i] for i in s)
                ):
                    continue

                yield s

    @staticmethod
    def get_cross(aligns):
        """ Get all crosses of a given list of alignments.

        :param aligns: a str representing alignments
        :return: the cross value for these alignments
        """

        tgt_idxs = [pair[1] for pair in aligns]

        return sum([1 for t1, t2 in combinations(tgt_idxs, 2) if t2 < t1])

    def _is_mwe(self, src_comb, tgt_comb):
        """ If all source items are connected to all target items, we assume that this is
            a segment that is likely to be a multi-word expression. """
        for src_idx in src_comb:
            if any(tgt_idx not in self._src2tgtlist_d[src_idx] for tgt_idx in tgt_comb):
                return False

        return True

def aligns_from_str(aligns):
    return sorted([tuple(map(int, align.split("-"))) for align in aligns.split()])

def aligns_to_str(aligns):
    """ Convert list of alignments (tuple of src, tgt) to GIZA/Pharaoh string """
    return " ".join([f"{s}-{t}" for s, t in aligns])
