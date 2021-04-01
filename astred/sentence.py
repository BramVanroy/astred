from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, TYPE_CHECKING, Union

from .utils import SPACY_AVAILABLE, STANZA_AVAILABLE, load_parser

if SPACY_AVAILABLE:
    from spacy.language import Language as SpacyLanguage
    from spacy.tokens.doc import Doc as SpacyDoc
    from spacy.tokens.span import Span as SpacySpan

if STANZA_AVAILABLE:
    from stanza.models.common.doc import Document as StanzaDoc
    from stanza.models.common.doc import Sentence as StanzaSentence
    from stanza.pipeline.core import Pipeline as StanzaPipeline

from .base import SpanMixin
from .enum import Side
from .span import Span
from .tree import Tree
from .word import Null, Word

logger = logging.getLogger("astred")

if TYPE_CHECKING:
    from .aligned import AlignedSentences


@dataclass(eq=False)
class Sentence(SpanMixin):
    words: List[Word] = field(default_factory=list, repr=False)
    side: Optional[Side] = field(default=None)

    tree: Tree = field(default=None, compare=False, repr=False, init=False)
    merged_tree: Tree = field(default=None, compare=False, repr=False, init=False)
    _aligned_sentence: Sentence = field(default=None, repr=False, init=False)
    aligned_sentences: AlignedSentences = field(default=None, repr=False, init=False)
    root: Word = field(default=None, repr=False, init=False)

    seq_spans: List[Span] = field(default_factory=list, compare=False, repr=False, init=False)
    sacr_spans: List[Span] = field(default_factory=list, compare=False, repr=False, init=False)

    _sentence: Union[StanzaSentence, SpacySpan] = field(default=None, repr=False)

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
    def aligned_sentence(self) -> Sentence:
        return self._aligned_sentence

    @aligned_sentence.setter
    def aligned_sentence(self, sentence: Sentence):
        self._aligned_sentence = sentence
        self.words = [Null()] + self.words
        self.attach_self_to_words()

    @property
    def word_cross(self) -> int:
        return self.aligned_sentences.word_cross if self.aligned_sentences else None

    @property
    def seq_cross(self) -> int:
        return self.aligned_sentences.seq_cross if self.aligned_sentences else None

    @property
    def sacr_cross(self) -> int:
        return self.aligned_sentences.sacr_cross if self.aligned_sentences else None

    @property
    def no_null_seq_spans(self) -> List[Span]:
        return [s for s in self.seq_spans if not s.is_null]

    @property
    def no_null_sacr_spans(self) -> List[Span]:
        return [s for s in self.sacr_spans if not s.is_null]

    def attach_self_to_words(self):
        for word in self.words:
            word.doc = self

    @staticmethod
    def _on_multiple_error_handling(sents, on_multiple: str = "raise") -> Optional[Union[StanzaSentence, SpacySpan]]:
        if len(sents) > 1:
            try:
                sents_repr = "\n".join([" ".join([w.text for w in sent]) for sent in sents])
            except TypeError:
                sents_repr = "\n".join([" ".join([w.text for w in sent.words]) for sent in sents])

            if on_multiple == "raise":
                raise ValueError(f"More than one sentence given, which is not allowed. If you wish to be more lenient,"
                                 f" you can set 'on_multiple' to warn or ignore.\nIn those cases only the first"
                                 f" sentence of those available will be used, but be aware that this may lead to"
                                 f" unexpected results.\nAlternatively, you can use the 'none' option, which will"
                                 f" return None if a parse contains more than one sentence."
                                 f"\n\nSentences:\n{sents_repr}")
            elif on_multiple == "warn":
                logger.warning("More than one sentence found in this parse. Will only use the first one."
                               f"\n\nSentences:\n{sents_repr}")
            elif on_multiple == "none":
                return None

        # At this point, either only one sent is given and we need the first one, or multiple ones are given and we
        # have warned the user about it.
        return sents[0]

    @classmethod
    def from_parser(cls, doc: Union[StanzaDoc, StanzaSentence, SpacyDoc, SpacySpan],
                    include_subtypes: bool = False,
                    on_multiple: str = "raise") -> Sentence:
        if on_multiple not in ("raise", "warn", "ignore", "none"):
            raise ValueError(f"'on_multiple' must be one of raise, warn, ignore ({on_multiple} given)")

        # If the given element is a full parsed doc, we need to check how many sentences it has (we only want one)
        if isinstance(doc, StanzaDoc):
            sentence = cls._on_multiple_error_handling(doc.sentences, on_multiple=on_multiple)
        elif isinstance(doc, SpacyDoc):
            sentence = cls._on_multiple_error_handling(list(doc.sents), on_multiple=on_multiple)
        else:
            # If it is a StanzaSentence or SpacySpan, we can just continue with that
            sentence = doc

        if isinstance(sentence, StanzaSentence):
            return cls([Word.from_stanza(w, include_subtypes=include_subtypes) for w in sentence.words],
                       _sentence=sentence)
        elif isinstance(sentence, SpacySpan):
            return cls([Word.from_spacy(w, include_subtypes=include_subtypes) for w in sentence], _sentence=sentence)

    @classmethod
    def from_text(cls, text: str,
                  nlp_or_model: Union[StanzaPipeline, SpacyLanguage, str],
                  parser: str = None,
                  is_tokenized: bool = True,
                  include_subtypes: bool = False,
                  on_multiple: str = "raise",
                  **kwargs) -> Sentence:
        if ((STANZA_AVAILABLE and isinstance(nlp_or_model, StanzaPipeline))
                or (SPACY_AVAILABLE and isinstance(nlp_or_model, SpacyLanguage))):
            return cls.from_parser(nlp_or_model(text), include_subtypes=include_subtypes, on_multiple=on_multiple)
        else:
            nlp = load_parser(nlp_or_model, parser, is_tokenized=is_tokenized, **kwargs)
            return cls.from_text(text, nlp, include_subtypes=include_subtypes,  on_multiple=on_multiple,)
