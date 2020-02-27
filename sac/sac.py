from collections import defaultdict

from . import Cross, aligns_to_str
from astred.GenericTree import GenericTree


class SAC(Cross):
    def __init__(self, alignments, src_text, tgt_text, src_lang='en', tgt_lang='nl'):
        super(SAC, self).__init__(alignments)
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
            pass
        return self._sac_aligns

    @property
    def sac_cross(self):
        if self._sac_cross is None:
            pass
        return self._sac_cross

    @property
    def sac_groups(self):
        if self._sac_groups is None:
            pass
        return self._sac_groups

    @classmethod
    def from_list(cls, align_list, *args, **kwargs):
        return cls(aligns_to_str(align_list),
                   *args,
                   **kwargs)

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

    def regroup_by_subtrees(self):
        #TODO: save new groups
        # test that all indices are present
        #
        modified_groups = []
        for group in self.groups:
            print('GROUP', group)
            if len(group) == 1:
                modified_groups.append(group)
                continue

            src_idxs, tgt_idxs = zip(*group)
            dir_idxs = {'src': src_idxs, 'tgt': tgt_idxs}
            dir_to_valid_idxs = defaultdict(list)
            for direction, idxs in dir_idxs.items():
                if len(set(idxs)) == 1:
                    continue

                dir_tree = getattr(self, f"{direction}_tree")
                grouped_per_level = dir_tree.grouped_per_level(idxs)
                # sort from deepest (highest number) to topmost
                grouped_per_level = {k: grouped_per_level[k] for k in sorted(grouped_per_level, reverse=True)}

                print(direction, 'grouped per level->parent_idx->word_idx')
                parent_to_idxs = defaultdict(list)
                found_parents = set()
                # Use the levels to start from the bottom (highest level number)
                for level, word_idxs in grouped_per_level.items():
                    for word_idx in word_idxs:
                        # Don't process a word if it is already a saved parent somewhere else
                        if word_idx in found_parents:
                            continue
                        parent_idx = dir_tree.word_order_idx_mapping[word_idx].parent().word_order_idx
                        # Only save if the parent_idx is also part of this group
                        # So in practice, only save children if their parent is also in the group
                        if parent_idx in idxs:
                            found_parents.add(parent_idx)
                            parent_to_idxs[parent_idx].append(word_idx)

                for k, v in parent_to_idxs.items():
                    dir_to_valid_idxs[direction].append([k] + v)

            dir_to_valid_idxs = dict(dir_to_valid_idxs)
            if dir_to_valid_idxs:
                group_d = dict(group)
                print('GROUP_D', group_d)
                print(dir_to_valid_idxs)
            else:
                pass

            # 1. create groups by finding items that are connected through the tree. All connections must go through
            # indices that are actually part of the group (do on both source and target side)
            # 2. start with biggest group on source, find their alignments, and check whether those all belong
            # to the same newly created group.
            # 2.1 if yes: add the source subgroup and tgt subgroup as a new group
            # 2.2 if no: find the largest subgroup in the aligned target items and align that with the new source group
            # add the cut-off target items as their own group, aligned with the new source group