from typing import List, Optional

import stanza

from . import word


class Sentence(list):
    def __init__(self, tokens: List, side: Optional[str] = None):
        super().__init__(tokens)
        if side not in [None, "src", "tgt"]:
            raise ValueError("'side' must be one of None, 'src', 'tgt'")
        self.side = side

    @classmethod
    def from_stanza(
        cls, sentence: stanza.models.common.doc.Sentence, side: Optional[str] = None
    ):
        return cls([word.Word.from_stanza(w) for w in sentence.words], side=side)
