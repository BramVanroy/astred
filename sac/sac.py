from collections import defaultdict

from . import Cross
from astred.GenericTree import GenericTree


class SAC(Cross):
    def __init__(self, alignments, src_text, tgt_text, src_lang='en', tgt_lang='nl'):
        super(SAC, self).__init__(alignments)
        self.src_text = src_text
        self.src_tokens = src_text.split()
        self.tgt_text = tgt_text
        self.tgt_tokens = tgt_text.split()

        self.src_tree = GenericTree.from_string(src_text, lang_or_model=src_lang)
        self.tgt_tree = GenericTree.from_string(tgt_text, lang_or_model=tgt_lang)

        self._sac_aligns = None
        self._sac_aligns_str = None
        self._sac_cross = None
        self._sac_groups = None

    @property
    def sac_aligns(self):
        if self._sac_aligns is None:
            pass
        return self._sac_aligns

    @property
    def sac_aligns_str(self):
        if self._sac_aligns_str is None:
            pass
        return self._sac_aligns_str

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
        return cls(cls.aligns_to_str(align_list),
                   *args,
                   **kwargs)

    def grouped_trees(self):
        #TODO: save new groups
        # test that all indices are present
        #
        modified_groups = []
        for group in self.groups:
            src_idxs, tgt_idxs = zip(*group)
            dir_idxs = {'src': src_idxs, 'tgt': tgt_idxs}

            for direction, idxs in dir_idxs.items():
                if len(set(idxs)) == 1:
                    continue

                dir_tree = getattr(self, f"{direction}_tree")
                grouped_per_level = dir_tree.grouped_per_level(idxs)
                # sort from deepest (highest number) to topmost
                grouped_per_level = {k: grouped_per_level[k] for k in sorted(grouped_per_level, reverse=True)}

                print(direction, 'grouped per level->parent_idx->word_idx')
                parent_to_level_to_idxs = {}
                found_parents = set()
                # Use the levels to start from the bottom (highest level number)
                for level, word_idxs in grouped_per_level.items():
                    parent_to_level_to_idxs[level] = defaultdict(list)
                    for word_idx in word_idxs:
                        # Don't process a word if it is already a saved parent somewhere else
                        if word_idx in found_parents:
                            continue
                        parent_idx = self.src_tree.word_order_idx_mapping[word_idx].parent().word_order_idx
                        # Only save if the parent_idx is also part of this group
                        # So in practice, only save children if their parent is also in the group
                        if parent_idx in idxs:
                            found_parents.add(parent_idx)
                            parent_to_level_to_idxs[level][parent_idx].append(word_idx)

                    parent_to_level_to_idxs[level] = dict(parent_to_level_to_idxs[level])

                print(parent_to_level_to_idxs)
