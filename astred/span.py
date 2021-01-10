from __future__ import annotations  # so that we can use the class in typing

from dataclasses import dataclass, field
from functools import cached_property
from typing import Dict, List, NamedTuple, Optional

from .base import Crossable, SpanMixin
from .enum import SpanType
from .tree import Tree
from .word import Null, Word


@dataclass
class Span(Crossable, SpanMixin):
    words: List[Word] = field(default_factory=list, repr=False)
    span_type: SpanType = None
    tree: Tree = field(default=None, init=False, repr=False)
    attach: bool = field(default=True)

    def __repr__(self):
        return f"{self.__class__.__name__}(id={self.id}, span_type={self.span_type}, text={self.text})"

    def __post_init__(self):
        super(Span, self).__post_init__()
        if not self.words:
            raise ValueError("'words' cannot be empty")
        elif len(self.words) == 1 and self.words[0].is_null and not self.is_null:
            raise ValueError(
                "The only given word for this span is a Null item but the span's 'is_null'"
                " is not set to True. Did you mean to initialize a NullSpan?"
            )

        if self.is_null and not isinstance(self, NullSpan):
            raise ValueError(
                f"Only {NullSpan.__name__} spans can be set to is_null=True"
            )

        if self.span_type is None or not isinstance(self.span_type, SpanType):
            raise ValueError(
                f"'span_type' must be one of {SpanType._member_names_} from {SpanType.__name__} enum"
            )

        if self.is_valid_subtree:
            self.tree = Tree.from_span(self, self.items_per_level[self.root_level][0], self.doc)

        if self.attach:
            self.attach_self_to_words()

    def get_word_by_doc_idx(self, idx) -> Word:
        if idx not in self.word_idxs:
            raise IndexError(f"This index ({idx}) does not exist in  {self}")

        return self.doc[idx]

    def attach_self_to_words(self):
        attr = f"{self.span_type}_group"

        for word_idx, word in enumerate(self):
            setattr(word, attr, self)
            setattr(word, f"id_in_{attr}", word_idx)

    @cached_property
    def items_per_level(self) -> Optional[Dict[int, List]]:
        # When some word in this span does not have its tree set, this should fail
        try:
            levels = set([word.tree.level for word in self])
        except AttributeError:
            return None

        return {
            level: [word for word in self if word.tree.level == level]
            for level in sorted(levels, reverse=True)
        }

    @cached_property
    def root_level(self) -> Optional[int]:
        if self.items_per_level:
            return min(self.items_per_level.keys())
        else:
            return None

    @cached_property
    def is_valid_subtree(self) -> Optional[bool]:
        """ valid subtrees need to all be connected. That means that
            for all nodes, their parents' idx (head) must be present except for the topmost level
            and that the topmost level can only contain one node (as the main ancestor) """
        # If any of these words do not have a tree set,
        # we won't even try to set a tree for this span nor check its validity
        # Also sets sets to None for NullSpans
        if any(not word.tree for word in self):
            return None

        if len(self) == 1:
            return True

        # A valid subtree can only have one top-level node
        if len(self.items_per_level[self.root_level]) > 1:
            return False

        return not any(
            w.head not in self.word_idxs
            for w in self
            if w.tree.level != self.root_level
        )

    @classmethod
    def sacr_from_seq(cls, span: Span, idx: int):
        return cls(id=idx, doc=span.doc, is_null=span.is_null, words=span.words)


class NullSpan(Span):
    def __init__(self, null_word: Null, span_type: SpanType = None):
        if not null_word.is_null:
            raise ValueError(
                "words inside a NullSpan need to be Null words and can only be one single word."
            )
        super().__init__(id=0, words=[null_word], span_type=span_type, is_null=True)
        self.seq_cross = None


SpanPair = NamedTuple("SpanPair", [("src", Span), ("tgt", Span), ("mwe", bool)])
