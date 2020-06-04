from functools import cached_property
from itertools import combinations
from typing import List, Set

from ..components.group import Group, SeqGroup
from .align import Align, AlignedIdxs
from .word_align import WordAlign


class SeqAlign(Align):
    def __init__(self, src: List[Group], tgt: List[Group], idxs: List[AlignedIdxs]):
        super().__init__(src, tgt, idxs)

    @classmethod
    def from_word_level(cls, word_align: WordAlign):
        groups = word_align.word_align_to_groups()

        src_groups = []
        tgt_groups = []
        for group_idx, group in enumerate(groups):
            src_group = SeqGroup([], side="src", idx=group_idx)
            tgt_group = SeqGroup([], side="tgt", idx=group_idx)
            for align in group:
                src_group.append(word_align.src[align.src])
                tgt_group.append(word_align.tgt[align.tgt])
            src_groups.append(src_group)
            tgt_groups.append(tgt_group)

        aligns = word_align.groups_to_align(groups)

        return cls(src_groups, tgt_groups, aligns)

    def align(self):
        for src_grp_idx, tgt_grp_idx in self:
            src_group = self.src[src_grp_idx]
            tgt_group = self.tgt[tgt_grp_idx]

            src_group.add_aligned_with(tgt_group)
            tgt_group.add_aligned_with(src_group)
