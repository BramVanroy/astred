from typing import List, Optional

from astred.components.group import Group
from astred.utils import AlignedIdxs


class SeqCross(Group):
    def __init__(self, words: List, side: Optional[str] = None):
        super().__init__(words, side)

        self.aligned_with: Optional[SeqCross] = None

    def add_aligned_with(self, sentence, alignments: AlignedIdxs):
        pass
