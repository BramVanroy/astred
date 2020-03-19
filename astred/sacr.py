from .cross import _Cross
from .tree import GenericTree
from .utils import AlignmentPair


class SACr(_Cross):
    def __init__(
        self,
        alignments,
        src_segment,
        tgt_segment,
        src_lang="en",
        tgt_lang="nl",
        use_gpu=True,
        **kwargs,
    ):
        super().__init__(alignments, **kwargs)
        self.src_segment = src_segment
        self.src_tokens = src_segment.split()
        self.tgt_segment = tgt_segment
        self.tgt_tokens = tgt_segment.split()

        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.use_gpu = use_gpu

        # for performance reasons, only init trees when we need them
        self._src_tree = None
        self._tgt_tree = None

        self._sacr_aligns = None
        self._sacr_cross = None
        self._sacr_groups = None
        self._sacr_mwe_groups = None

    @property
    def sacr_aligns(self):
        if self._sacr_aligns is None:
            self._sacr_aligns = self._groups_to_align(self.sacr_groups)
        return self._sacr_aligns

    @property
    def sacr_cross(self):
        if self._sacr_cross is None:
            self._sacr_cross = self._get_cross(self.sacr_aligns)
        return self._sacr_cross

    @property
    def sacr_groups(self):
        if self._sacr_groups is None:
            self._sacr_groups, self._sacr_mwe_groups = self._regroup_by_subtrees()
        return self._sacr_groups

    @property
    def src_tree(self):
        if self._src_tree is None:
            self._src_tree = GenericTree.from_string(
                self.src_segment, lang_or_model=self.src_lang, use_gpu=self.use_gpu
            )
        return self._src_tree

    @property
    def tgt_tree(self):
        if self._tgt_tree is None:
            self._tgt_tree = GenericTree.from_string(
                self.tgt_segment, lang_or_model=self.tgt_lang, use_gpu=self.use_gpu
            )
        return self._tgt_tree

    def _get_null_aligns(self):
        """ Get missing idxs (= null alignments) and return them as alignments to -1.
            We use -1 so that we can still order our lists containing null alignments.
            Expects SORTED input lists.
            Overwrites the implementation in _Cross because now we know the actual number
            of tokens from the given text, and so we can catch when the last token(s) are not aligned
        """
        src_missing = [
            AlignmentPair(idx, -1)
            for idx in range(len(self.src_tokens))
            if idx not in self.src_idxs
        ]
        tgt_missing = [
            AlignmentPair(-1, idx)
            for idx in range(len(self.tgt_tokens))
            if idx not in self.tgt_idxs
        ]

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

    def _regroup_by_subtrees(self):
        mwe_groups = []
        modified_groups = []
        src_idxs_grouped = set()
        tgt_idxs_grouped = set()

        # we don't have to check for internal crosses here,
        # because these groups are already selected for that
        for group in self.seq_groups:
            # if the group only consists of one src-tgt pair (one tuple)
            # just add it and continue
            if len(group) == 1:
                src_idxs_grouped.add(group[0].src)
                tgt_idxs_grouped.add(group[0].tgt)
                modified_groups.append(group)
                continue

            group_src_idxs, group_tgt_idxs = self._get_src_tgt_idxs(group)

            src_combs = self._consec_combinations(group_src_idxs)
            # need to listify the generator because it is in the internal loop
            # the generator would exhaust after the first outer loop, but we need to re-use
            # it for all outer loops, so build a list
            tgt_combs = list(self._consec_combinations(group_tgt_idxs))

            # Try grouping a src_comb with a tgt_comb
            for src_comb in src_combs:
                # If any item in this combination has already been grouped in another
                # combo group, continue
                if any(src in src_idxs_grouped for src in src_comb):
                    continue
                n_src_items = len(src_comb)

                for tgt_comb in tgt_combs:
                    # If any item in this combination has already been grouped in another
                    # combo group, continue
                    if any(tgt in tgt_idxs_grouped for tgt in tgt_comb):
                        continue

                    n_tgt_items = len(tgt_comb)
                    is_mwe = False
                    if n_src_items > 1 and n_tgt_items > 1:
                        is_mwe = self.group_mwe and self._is_in_mwe(src_comb, tgt_comb)

                    # if a combination is a valid MWE, don't split them up based on subtrees
                    if is_mwe or (
                        not self._has_external_aligns(src_comb, tgt_comb)
                        and self._is_valid_subtree(src_comb, "src")
                        and self._is_valid_subtree(tgt_comb, "tgt")
                    ):
                        # Keep track of src+tgt idxs that are already grouped
                        src_idxs_grouped.update(src_comb)
                        tgt_idxs_grouped.update(tgt_comb)
                        # Get all alignments of this group and add them as group
                        alignments_of_group = sorted(
                            [i for src in src_comb for i in self.src2aligns_d[src]]
                        )
                        modified_groups.append(alignments_of_group)
                        if is_mwe:
                            mwe_groups.append(alignments_of_group)
                        # Break because we have found a suitable group
                        break

        modified_groups = self._add_unsolved_idxs(
            src_idxs_grouped, tgt_idxs_grouped, modified_groups
        )

        return sorted(modified_groups), mwe_groups
