from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, NamedTuple, TYPE_CHECKING

from .base import Crossable
from .utils import SPACY_AVAILABLE, STANZA_AVAILABLE

if TYPE_CHECKING:
    from .span import Span, SpanPair
    from .tree import Tree

    if SPACY_AVAILABLE:
        from spacy.tokens.token import Token as SpacyToken

    if STANZA_AVAILABLE:
        from stanza.models.common.doc import Word as StanzaWord


@dataclass(repr=False)
class Word(Crossable):
    text: str = field(repr=False, default=None)
    lemma: str = field(repr=False, default=None)
    head: int = field(repr=False, default=None)
    deprel: str = field(repr=False, default=None)
    upos: str = field(repr=False, default=None)
    xpos: str = field(repr=False, default=None)
    feats: str = field(repr=False, default=None)

    seq_group: Span = field(default=None, init=False, compare=False, repr=False)
    id_in_seq_group: int = field(default=None, init=False, compare=False, repr=False)

    sacr_group: Span = field(default=None, init=False, compare=False, repr=False)
    id_in_sacr_group: int = field(default=None, init=False, compare=False, repr=False)

    tree: Tree = field(default=None, init=False, compare=False, repr=False)
    connected: List[Word] = field(default_factory=list, init=False, repr=False)
    connected_repr: str = field(default=None, init=False, repr=False)

    _word: Any = field(default=None, repr=False)

    @property
    def is_root(self) -> bool:
        return self.head == 0

    @property
    def is_root_in_sacr_group(self) -> bool:
        return self.sacr_group.root is self

    def __post_init__(self):
        super(Word, self).__post_init__()
        if self.is_null and not isinstance(self, Null):
            raise ValueError(f"Only {Null.__name__} words can be set to is_null=True")
        elif not self.is_null and isinstance(self, Null):
            raise ValueError(f"{Null.__name__} words must be set to is_null=True")

    def changes(self, attr: str = "deprel") -> Dict[int, bool]:
        attr_val = getattr(self, attr)
        return (
            {word.id: attr_val != getattr(word, attr) for word in self.aligned if not word.is_null}
            if attr_val is not None
            else None
        )

    def num_changes(self, attr: str = "deprel") -> int:
        # `changes()` is a dict of int, bool but summing works due to implicit casting
        changes = self.changes(attr)
        return sum(changes.values()) if changes else None

    def avg_num_changes(self, attr: str = "deprel") -> float:
        changes = self.changes(attr)
        return (sum(changes.values()) / len(changes)) if changes else None

    @classmethod
    def from_spacy(cls, word: SpacyToken, include_subtypes: bool = False):
        # Spacy starts counting at 0, so +1.
        return cls(
            id=word.i + 1,
            text=word.text,
            lemma=word.lemma_,
            head=0 if word.head.i == word.i else word.head.i + 1,  # the root node is its own head, so check
            deprel=word.dep_ if include_subtypes else word.dep_.split(":")[0],
            upos=word.pos_,
            xpos=word.tag_,
            feats=word.morph if word.morph else "_",
            _word=word
        )

    @classmethod
    def from_stanza(cls, word: StanzaWord, include_subtypes: bool = False):
        # Stanza starts counting at 1 (0 reserved for a ROOT node). We also start at 1 so that 0 can be used for Null
        return cls(
            id=int(word.id),
            text=word.text,
            lemma=word.lemma,
            head=int(word.head),
            deprel=word.deprel if include_subtypes else word.deprel.split(":")[0],
            upos=word.upos,
            xpos=word.xpos,
            feats=word.feats if word.feats else "_",
            _word=word
        )


class Null(Word):
    def __init__(self):
        super().__init__(id=0, text="[[NULL]]", is_null=True)

    def changes(self, attr: str = "deprel") -> None:
        return None

    def num_changes(self, attr: str = "deprel") -> None:
        return None

    def avg_num_changes(self, attr: str = "deprel") -> None:
        return None


WordPair = NamedTuple("WordPair", [("src", Word), ("tgt", Word)])


def spanpair_to_wordpairs(spanpair: SpanPair) -> List[WordPair]:
    wpairs = []
    for word in spanpair.src:
        for aligned_w in word.aligned:
            if aligned_w in spanpair.tgt:
                wpairs.append(WordPair(word, aligned_w))
    return wpairs
