from typing import Generator, List, Optional, Union


try:
    import stanza
    from stanza import Pipeline as StanzaPipeline

    STANZA_AVAILABLE = True
except (ImportError, AttributeError):
    STANZA_AVAILABLE = False

try:
    import spacy
    from spacy.language import Language as SpacyLanguage
    from spacy.tokens import Doc as SpacyDoc
    from spacy.vocab import Vocab as SpacyVocab

    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

if SPACY_AVAILABLE:

    class SpacyPretokenizedTokenizer:
        """Custom tokenizer to be used in spaCy when the text is already pretokenized."""

        def __init__(self, vocab: SpacyVocab):
            """Initialize tokenizer with a given vocab
            :param vocab: an existing vocabulary (see https://spacy.io/api/vocab)
            """
            self.vocab = vocab

        def __call__(self, inp: Union[List[str], str]) -> SpacyDoc:
            """Call the tokenizer on input `inp`.
            :param inp: either a string to be split on whitespace, or a list of tokens
            :return: the created Doc object
            """
            if isinstance(inp, str):
                words = inp.split()
                spaces = [True] * (len(words) - 1) + ([True] if inp[-1].isspace() else [False])
                return SpacyDoc(self.vocab, words=words, spaces=spaces)
            elif isinstance(inp, list):
                return SpacyDoc(self.vocab, words=inp)
            else:
                raise ValueError(
                    "Unexpected input format. Expected string to be split on whitespace, or list of tokens."
                )

    @SpacyLanguage.component("prevent_sbd")
    def spacy_prevent_sbd(doc: SpacyDoc):
        """Disables spaCy's sentence boundary detection."""
        for token in doc:
            token.is_sent_start = False
        return doc


def unique_list(groups: List):
    """Filter list of lists so that:
    - the sublists only contain unique items (no duplicates);
    - the sublists themselves are unique (two identical sublists cannot exists)"""

    def unique(main_list: List):
        uniq = []
        uniq_ids = set()
        for item in main_list:
            is_list = isinstance(item, list)
            item = [item] if not is_list else item
            item_repr = tuple([f"{i.doc.side if i.doc else 'none'}-{i.id}" for i in item])
            if item_repr not in uniq_ids:
                uniq.append(item[0] if not is_list else item)
                uniq_ids.add(item_repr)
        return uniq

    if isinstance(groups[0], list):
        # Make sure that items in sublists are unique
        groups = [unique(group) for group in groups]

    # Make sure that sublists themselves are unique
    return unique(groups)


def rebase_to_idxs(idxs: List[int]):
    """Convert values to indices. This ensure that there are no strange gaps
    between sequence alignments (e.g. when an index is not word-aligned)"""
    l_sort = sorted(list(set(idxs)))

    return [l_sort.index(x) for x in idxs]


def pair_combs(all_pairs: List, min_length: int = 2) -> Generator[List, None, None]:
    n_pairs = len(all_pairs)
    for i in range(n_pairs, min_length - 1, -1):
        for j in range(n_pairs - i + 1):
            pairs = all_pairs[j : j + i]
            if any(item.is_null for pair in pairs for item in pair):
                continue
            yield pairs


def load_parser(
    model_or_lang: str,
    parser: Optional[str] = None,
    *,
    auto_download: bool = True,
    is_tokenized: bool = True,
    use_gpu: bool = True,
    **kwargs,
):
    try:
        if parser == "spacy":
            if use_gpu:
                spacy.prefer_gpu()  # Only use GPU if it is available
            else:
                spacy.require_cpu()

            if is_tokenized:
                # Disable sentence segmentation through senter or sentencizer component as well
                nlp = spacy.load(model_or_lang, exclude=["senter", "sentencizer"], **kwargs)
                nlp.tokenizer = SpacyPretokenizedTokenizer(nlp.vocab)
                # It is still possible that the dependency parser leads to segmentation, disable
                nlp.add_pipe("prevent_sbd", name="prevent-sbd", before="parser")
            else:
                nlp = spacy.load(model_or_lang, **kwargs)
        elif parser == "stanza":
            if auto_download:
                stanza.download(model_or_lang, verbose=False)
            nlp = StanzaPipeline(
                processors="tokenize,pos,lemma,depparse",
                lang=model_or_lang,
                tokenize_pretokenized=is_tokenized,
                use_gpu=use_gpu,
                logging_level="WARNING",
                **kwargs,
            )
        else:
            if STANZA_AVAILABLE:
                return load_parser(
                    model_or_lang, parser="stanza", is_tokenized=is_tokenized, use_gpu=use_gpu, **kwargs
                )
            elif SPACY_AVAILABLE:
                return load_parser(model_or_lang, parser="spacy", is_tokenized=is_tokenized, use_gpu=use_gpu, **kwargs)
            else:
                raise ImportError
    except (NameError, ImportError):
        err = "Stanza or spaCy not installed so cannot instantiate a parser"
        err += f" ({parser} requested)" if parser else ""
        raise ImportError(err)

    return nlp


try:
    from functools import cached_property
except (ImportError, AttributeError):

    class cached_property(property):
        """
        Descriptor that mimics @property but caches output in member variable.
        From tensorflow_datasets
        Built-in in functools from Python 3.8.
        """

        def __get__(self, obj, objtype=None):
            # See docs.python.org/3/howto/descriptor.html#properties
            if obj is None:
                return self
            if self.fget is None:
                raise AttributeError("unreadable attribute")
            attr = "__cached_" + self.fget.__name__
            cached = getattr(obj, attr, None)
            if cached is None:
                cached = self.fget(obj)
                setattr(obj, attr, cached)
            return cached
