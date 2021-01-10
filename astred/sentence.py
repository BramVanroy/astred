from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Union, Any

from stanza.models.common.doc import Document as StanzaDoc
from stanza.models.common.doc import Sentence as StanzaSentence
from stanza.pipeline.core import Pipeline as StanzaPipeline

from .base import SpanMixin
from .enum import Side
from .span import Span
from .tree import Tree
from .utils import load_nlp
from .word import Null, Word

logger = logging.getLogger("astred")


@dataclass(eq=False)
class Sentence(SpanMixin):
    words: List[Word] = field(default_factory=list, repr=False)
    side: Optional[Side] = field(default=None)

    tree: Tree = field(default=None, compare=False, repr=False, init=False)
    merged_tree: Tree = field(default=None, compare=False, repr=False, init=False)
    _aligned_sentence: Sentence = field(default=None, repr=False, init=False)
    aligned_sentences: Any = field(default=None, repr=False, init=False)
    root: Word = field(default=None, repr=False, init=False)

    seq_spans: List[Span] = field(
        default_factory=list, compare=False, repr=False, init=False
    )
    sacr_spans: List[Span] = field(
        default_factory=list, compare=False, repr=False, init=False
    )

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(text={self.text}, side={self.side},"
            f" root={self.root.text if self.root else None})"
        )

    def __post_init__(self):
        self.attach_self_to_words()
        roots = [w for w in self if w.is_root]

        if len(roots) != 1:
            logger.warning(
                "Can only create a tree for a sentence if it has one (and only one) root (Word.is_root)."
                " A tree for this sentence was not created."
            )
        elif not all(w.head is not None for w in self if not w.is_root or w.is_null):
            logger.warning(
                "Can only create a tree for a sentence if all of its words (except potential Null and the root of the"
                " sentence) have a 'head' attribute set. A tree for this sentence was not created."
            )
        else:
            self.tree = Tree.from_sentence(self)
            self.root = roots[0]

    @property
    def aligned_sentence(self):
        return self._aligned_sentence

    @aligned_sentence.setter
    def aligned_sentence(self, sentence: Sentence):
        self._aligned_sentence = sentence
        self.words = [Null()] + self.words
        self.attach_self_to_words()

    @property
    def no_null_seq_spans(self):
        return [s for s in self.seq_spans if not s.is_null]

    @property
    def no_null_sacr_spans(self):
        return [s for s in self.sacr_spans if not s.is_null]

    def attach_self_to_words(self):
        for word in self.words:
            word.doc = self

    @classmethod
    def from_stanza(cls, doc: Union[StanzaDoc, StanzaSentence]):
        # doc can either be a Sentence or a Doc. In the latter case we only use the first sentence in it.
        if isinstance(doc, StanzaDoc):
            if len(doc.sentences) > 1:
                logger.warning(
                    "More than one sentence found in this stanza parse. Will only use the first once."
                )
            sentence = doc.sentences[0]
        else:
            sentence = doc

        # Stanza starts counting at 1 (0 reserved for a ROOT node). We also start at 1 so that 0 can be used for Null
        return cls(
            [
                Word(
                    id=int(w.id),
                    text=w.text,
                    lemma=w.lemma,
                    head=int(w.head),
                    deprel=w.deprel,
                    upos=w.upos,
                    xpos=w.xpos,
                    feats=w.feats if w.feats else "_",
                )
                for w in sentence.words
            ]
        )

    @classmethod
    def from_text(
        cls, text: str, nlp_or_lang: Union[StanzaPipeline, str], **kwargs
    ):
        if isinstance(nlp_or_lang, StanzaPipeline):
            return cls.from_stanza(nlp_or_lang(text))
        else:
            nlp = load_nlp(nlp_or_lang, **kwargs)
            return cls.from_stanza(nlp(text))
