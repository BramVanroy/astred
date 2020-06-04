from typing import List, Optional

from astred.align.utils import AlignedIdxs


class Group(list):
    def __init__(self, words: List, side: Optional[str] = None, idx: Optional[int] = None):
        super().__init__(words)
        if side not in [None, "src", "tgt"]:
            raise ValueError("'side' must be one of None, 'src', 'tgt'")
        self.side: str = side
        self.idx = idx
        self.aligned_with: Optional[Group] = None

        self.aligned_idxs: List[int] = []
        self.aligned_groups: List = []
        self.moved_dists: List[int] = []

        self.is_aligned: bool = False

        self.cross: int = 0
        self.cross_aligned_idxs: List[int] = []

    def add_aligned_with(self, group):
        if not isinstance(group, self.__class__):
            raise ValueError(
                f"'group' ({group.__class__}) must be the same class as self ({self.__class__})"
            )
        self.is_aligned = True

        self.aligned_idxs.append(group.idx)
        self.aligned_groups.append(group)
        self.moved_dists.append(group.idx - self.idx)

    def add_cross(self, aligned_idx):
        self.cross += 1
        self.cross_aligned_idxs.append(aligned_idx)


class SeqGroup(Group):
    def __init__(self, words: List, side: Optional[str] = None, idx: Optional[int] = None):
        super().__init__(words, side, idx)
        self.set_word_group_idx()

    def set_word_group_idx(self):
        if self.idx is not None:
            for word_idx, word in enumerate(self):
                word.seq_group_idx = self.idx
                word.idx_in_seq_group = word_idx
