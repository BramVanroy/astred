from functools import cached_property
from typing import List, Union

from astred.align.seq_align import SeqAlign
from astred.align.utils import aligns_from_str
from astred.align.word_align import WordAlign


class AlignedSentences:
    def __init__(self, src, tgt, alignments: Union[List, str]):
        if isinstance(alignments, str):
            alignments = aligns_from_str(alignments)

        self.src = src
        self.tgt = tgt

        self.word_level = WordAlign(src, tgt, alignments)
        self.seq_level = SeqAlign.from_word_level(self.word_level)

    @cached_property
    def word_cross(self) -> int:
        return self.word_level.cross

    @cached_property
    def seq_cross(self) -> int:
        return self.seq_level.cross
