import logging
from functools import cached_property
from typing import List, Optional

import stanza

from astred.align.utils import AlignedIdxs
from astred.components import word
from astred.components.group import Group
from astred.utils import load_nlp


class Sentence(Group):
    def __init__(self, words: List, side: Optional[str] = None):
        super().__init__(words, side)

    @cached_property
    def word_cross(self):
        return sum([w.word_cross for w in self])

    def add_aligned_with(self, sentence, alignments: List[AlignedIdxs]):
        for src_idx, tgt_idx in alignments:
            # Do not align to "unaligned" indices
            if src_idx == -1 or tgt_idx == -1:
                continue

            if self.side == "src":
                self[src_idx].add_aligned_with(sentence[tgt_idx])
            else:
                self[tgt_idx].add_aligned_with(sentence[src_idx])

        self.aligned_with = sentence

    @classmethod
    def from_stanza(
        cls, sentence: stanza.models.common.doc.Sentence, side: Optional[str] = None
    ):
        return cls([word.Word.from_stanza(w) for w in sentence.words], side=side)

    @classmethod
    def from_text(cls, sentence: str, lang: str = "en", side: Optional[str] = None):
        nlp = load_nlp(lang)
        parsed = nlp(sentence)

        if len(parsed.sentences) > 1:
            logging.warning(
                f"'{sentence}' consists of more than one sentence. Using first sentence."
            )

        return cls.from_stanza(parsed.sentences[0], side=side)

    def __repr__(self):
        return f"<Sentence words={super().__repr__()} side={self.side}>"
