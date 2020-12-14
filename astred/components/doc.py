import logging
from dataclasses import dataclass, Field, field
from typing import List, Union

import stanza

from astred.components.word import Word


logger = logging.getLogger("astred")


@dataclass(init=False)
class Doc:
    words: List

    def __init__(self, words):
        if not isinstance(words, List):
            raise ValueError("Must be constructed based on a list of Words.")
        self.words = words
        self.process_words()

    def __getitem__(self, idx):
        return self.words[idx]

    def __iter__(self):
        return iter(self.words)

    def __len__(self):
        return len(self.words)

    @classmethod
    def from_stanza(cls, doc: Union[stanza.models.common.doc.Document, stanza.models.common.doc.Sentence]):
        if isinstance(doc, stanza.models.common.doc.Document):
            if len(doc.sentences) > 1:
                logger.warning("More than one sentence found in this stanza parse. Will only use the first once.")
            sent = doc.sentences[0]
        else:
            sent = doc

        # Stanza starts counting at 1 (0 reserved for a ROOT node). For convenience, we start at 0.
        return cls([Word(w.text,
                         int(w.id) - 1,
                         int(w.head) - 1,
                         w.deprel,
                         w.upos,
                         w.xpos,
                         w.feats if w.feats else "_") for w in sent.words])

    @classmethod
    def from_text(cls, text: str, nlp: stanza.pipeline.core.Pipeline):
        return cls.from_stanza(nlp(text))

    def process_words(self):
        # attach parent doc to all words
        for w in self.words:
            w.doc = self

