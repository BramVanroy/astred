from typing import Optional

import stanza


class Word:
    def __init__(
        self,
        text: str,
        udep: str,
        idx: int,
        head_idx: int,
        upos: Optional[str] = None,
        is_root: bool = False,
        parent_sentence=None,
    ):
        self.text = text
        self.udep = udep
        self.idx = idx
        self.head_idx = head_idx
        self.upos = upos
        self.is_root = is_root
        self.sentence = parent_sentence

        self.idx_seq_group: Optional[int] = None
        self.idx_sacr_group: Optional[int] = None

        self.word_cross: Optional[int] = None
        self.seq_cross: Optional[int] = None
        self.sacr_cross: Optional[int] = None

    @classmethod
    def from_stanza(
        cls, word: stanza.models.common.doc.Word, is_root: bool = False, parent_sentence=None,
    ):
        return cls(
            word.text,
            word.deprel,
            word.id,
            word.head,
            upos=word.upos if word.upos else None,
            is_root=is_root,
            parent_sentence=parent_sentence,
        )

    def __repr__(self):
        return (
            f"<Word text={self.text} udp={self.udep} idx={self.idx} head_idx={self.head_idx}>"
        )
