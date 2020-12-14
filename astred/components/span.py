from dataclasses import dataclass
from typing import List

from .doc import Doc
from .word import Word


@dataclass(init=False)
class Span:
    doc: Doc
    start: int
    end: int
    words: List[Word]

    def __init__(self, doc, start, end):
        if not (0 <= start <= end <= len(doc)):
            raise IndexError(f"start {start} or end {end} are less than zero or larger than the number of words"
                             f" in the doc {len(doc)} ")
        self.doc = doc
        self.start = start
        self.end = end
        self.words = self.doc[self.start:self.end]
        self._idxs = list(range(self.start, self.end))
