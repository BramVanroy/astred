from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, NamedTuple

from .base import Crossable


@dataclass(repr=False)
class Word(Crossable):
    text: str = field(repr=False, default=None)
    lemma: str = field(repr=False, default=None)
    head: int = field(repr=False, default=None)
    deprel: str = field(repr=False, default=None)
    upos: str = field(repr=False, default=None)
    xpos: str = field(repr=False, default=None)
    feats: str = field(repr=False, default=None)

    seq_group: Any = field(default=None, init=False, compare=False, repr=False)
    id_in_seq_group: int = field(default=None, init=False, compare=False, repr=False)

    sacr_group: Any = field(default=None, init=False, compare=False, repr=False)
    id_in_sacr_group: int = field(default=None, init=False, compare=False, repr=False)

    tree: Any = field(default=None, init=False, compare=False, repr=False)
    connected: List[Word] = field(default_factory=list, init=False, repr=False)
    connected_repr: str = field(default=None, init=False, repr=False)

    @property
    def is_root(self):
        return self.head == 0

    @property
    def is_root_in_sacr_group(self):
        return self.sacr_group.root is self

    def __post_init__(self):
        super(Word, self).__post_init__()
        if self.is_null and not isinstance(self, Null):
            raise ValueError(f"Only {Null.__name__} words can be set to is_null=True")
        elif not self.is_null and isinstance(self, Null):
            raise ValueError(f"{Null.__name__} words must be set to is_null=True")

        # Do not include UD subtypes. Only focus on the main types.
        self.deprel = self.deprel.split(":")[0] if self.deprel else self.deprel

    def changes(self, attr="deprel") -> Dict[int, bool]:
        attr_val = getattr(self, attr)
        return {word.id: attr_val != getattr(word, attr) for word in self.aligned if not word.is_null}

    def num_changes(self, attr="deprel") -> int:
        # `changes()` is a dict of int, bool but summing works due to implicit casting
        changes = self.changes(attr)
        return sum(changes.values()) if changes else None

    def avg_num_changes(self, attr="deprel") -> float:
        changes = self.changes(attr)
        return (sum(changes.values()) / len(changes)) if changes else None


class Null(Word):
    def __init__(self):
        super().__init__(id=0, text="[[NULL]]", is_null=True)

    def changes(self, attr="deprel") -> None:
        return None

    def num_changes(self, attr="deprel") -> None:
        return None

    def avg_num_changes(self, attr="deprel") -> None:
        return None


WordPair = NamedTuple("WordPair", [("src", Word), ("tgt", Word)])


def spanpair_to_wordpairs(spanpair) -> List[WordPair]:
    wpairs = []
    for word in spanpair.src:
        for aligned_w in word.aligned:
            if aligned_w in spanpair.tgt:
                wpairs.append(WordPair(word, aligned_w))
    return wpairs
