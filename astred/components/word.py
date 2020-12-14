from __future__ import annotations  # so that we can use the class in typing

from dataclasses import dataclass
from functools import cached_property


@dataclass
class Word:
    text: str
    idx: int
    head_idx: int
    deprel: str
    upos: str
    xpos: str
    feats: str
    _doc = None

    @property
    def doc(self):
        return self._doc

    @doc.setter
    def doc(self, doc):
        self._doc = doc

