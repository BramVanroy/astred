from typing import List, Optional

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
        self.text: str = text
        self.udep: str = udep
        # Note: idx starts counting at 1 because 0 is the invisible root in stanza
        self.idx: int = idx - 1
        self.head_idx: int = head_idx - 1

        self.upos: str = upos
        self.is_root: bool = is_root
        self.sentence = parent_sentence
        self.aligned_words: List[Word] = []

        self.aligned_idxs: List[int] = []
        self.moved_dists: List[int] = []

        self.is_aligned: bool = False

        self.seq_group_idx: Optional[int] = None
        self.idx_in_seq_group: Optional[int] = None
        self.sacr_group_idx: Optional[int] = None
        self.idx_in_sacr_group: Optional[int] = None

        self.word_cross: int = 0
        self.word_cross_aligned_idxs: List[int] = []

        self.seq_cross: int = 0
        self.sacr_cross: int = 0

    @property
    def aligned(self):
        return list(zip(*[self.aligned_idxs, self.aligned_words, self.moved_dists]))

    def add_aligned_with(self, word):
        self.is_aligned = True

        self.aligned_idxs.append(word.idx)
        self.aligned_words.append(word)
        self.moved_dists.append(word.idx - self.idx)

    def add_cross(self, aligned_idx):
        self.word_cross += 1
        self.word_cross_aligned_idxs.append(aligned_idx)

    @classmethod
    def from_stanza(
        cls, word: stanza.models.common.doc.Word, is_root: bool = False, parent_sentence=None,
    ):
        return cls(
            word.text,
            word.deprel,
            int(word.id),
            int(word.head),
            upos=word.upos if word.upos else None,
            is_root=is_root,
            parent_sentence=parent_sentence,
        )

    def __repr__(self):
        if self.is_aligned:
            return (
                f"<Word text={self.text} udp={self.udep} idx={self.idx} head_idx={self.head_idx}"
                f" is_aligned=True aligned_idxs={self.aligned_idxs}>"
            )
        else:
            return f"<Word text={self.text} udp={self.udep} idx={self.idx} head_idx={self.head_idx} is_aligned=False>"
