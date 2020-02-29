from astred import GenericTree

from . import Cross, aligns_to_str


class SAC(Cross):
    def __init__(self, alignments, src_text, tgt_text, src_lang="en", tgt_lang="nl", **kwargs):
        super(SAC, self).__init__(alignments, **kwargs)
        self.src_text = src_text
        self.src_tokens = src_text.split()
        self.n_src_tokens = len(self.src_tokens)
        self.tgt_text = tgt_text
        self.tgt_tokens = tgt_text.split()
        self.n_tgt_tokens = len(self.tgt_tokens)

        self.src_tree = GenericTree.from_string(src_text, lang_or_model=src_lang)
        self.tgt_tree = GenericTree.from_string(tgt_text, lang_or_model=tgt_lang)

        self._sac_aligns = None
        self._sac_cross = None
        self._sac_groups = None

    @property
    def sac_aligns(self):
        if self._sac_aligns is None:
            self._sac_aligns = self._groups_to_align(self.sac_groups)
        return self._sac_aligns

    @property
    def sac_cross(self):
        if self._sac_cross is None:
            self._sac_cross = self.get_cross(self.sac_aligns)
        return self._sac_cross

    @property
    def sac_groups(self):
        if self._sac_groups is None:
            self._sac_groups = self.regroup_by_subtrees()
        return self._sac_groups

    @classmethod
    def from_list(cls, align_list, *args, **kwargs):
        return cls(aligns_to_str(align_list), *args, **kwargs)

    def _get_null_aligns(self):
        """ Get missing idxs (= null alignments) and return them as alignments to -1.
            We use -1 so that we can still order our lists containing null alignments.
            Expects SORTED input lists.
            Overwrites naive implementation in cross which doesn't take actual n_tokens
            into account.
        """

        def missing_idxs(idxs, n_max):
            idxs = set(idxs)
            return [i for i in range(n_max) if i not in idxs]

        src_missing = [(idx, -1) for idx in missing_idxs(self.src_idxs, self.n_src_tokens)]
        tgt_missing = [(-1, idx) for idx in missing_idxs(self.tgt_idxs, self.n_tgt_tokens)]

        return src_missing + tgt_missing

    def _is_valid_subtree(self, idxs, direction):
        """ valid subtrees need to all be connected. That means that
            for all nodes, their parent_idx must be present except for the topmost level
            and that the topmost level can only contain one node (as the main ancestor) """
        dir_tree = getattr(self, f"{direction}_tree")
        grouped_per_level = dir_tree.grouped_per_level(idxs)

        # topmost level can only contain one node
        topmost_level = min(grouped_per_level.keys())
        if len(grouped_per_level[topmost_level]) > 1:
            return False
        # the topmost node's parent does not (cannot) need to be included
        del grouped_per_level[topmost_level]

        # sort from deepest (highest number) to topmost
        grouped_per_level = {
            k: grouped_per_level[k] for k in sorted(grouped_per_level, reverse=True)
        }

        for word_idxs in grouped_per_level.values():
            for word_idx in word_idxs:
                parent_idx = dir_tree.word_order_idx_mapping[word_idx].parent().word_order_idx
                if parent_idx not in idxs:
                    return False

        return True

    def _valid_subtree_combs(self, idxs, direction):
        """ Return sorted by largest possible combinations/subtree first."""
        idxs.sort()
        n_idxs = len(idxs)
        for i in range(n_idxs, 0, -1):
            for j in range(n_idxs - i + 1):
                r = idxs[j:j + i]
                if self._is_valid_subtree(r, direction):
                    yield r

    def regroup_by_subtrees(self):
        modified_groups = []
        src_idxs_grouped = set()
        tgt_idxs_grouped = set()

        for group in self.seq_groups:
            # if the group only consists of one src-tgt pair (one tuple)
            # just add it and continue
            if len(group) == 1:
                modified_groups.append(group)
                continue

            group_src_idxs, group_tgt_idxs = zip(*group)
            group_src_idxs = list(set(group_src_idxs))
            group_tgt_idxs = list(set(group_tgt_idxs))
            # src_combs and tgt_combs are only valid subtrees, so
            # not ALL combinations
            src_combs = self._valid_subtree_combs(group_src_idxs, "src")
            tgt_combs = self._valid_subtree_combs(group_tgt_idxs, "tgt")

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
                    # check that this pair of combs does not have external aligns
                    has_external_aligns = self._has_external_aligns(
                        src_comb, tgt_comb
                    )

                    if not has_external_aligns:
                        # Keep track of src+tgt idxs that are already grouped
                        src_idxs_grouped.update(src_comb)
                        tgt_idxs_grouped.update(tgt_comb)
                        # Get all alignments of this group and add them as group
                        alignments_of_group = [
                            i for src in src_comb for i in self.src2aligns_d[src]
                        ]
                        modified_groups.append(alignments_of_group)
                        # Break because we have found a suitable group
                        break

        modified_groups = self._add_unsolved_idxs(src_idxs_grouped,
                                                  tgt_idxs_grouped,
                                                  modified_groups)

        return sorted(modified_groups)
