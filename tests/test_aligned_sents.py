from pytest_cases import parametrize_with_cases

from astred import Null
from astred.enum import SpanType

from .conftest import TestAlignedSents


@parametrize_with_cases("aligned", cases=TestAlignedSents)
def test_aligned_sents__sents_attached(aligned):
    assert aligned.src.aligned_sentence == aligned.tgt
    assert aligned.tgt.aligned_sentence == aligned.src


@parametrize_with_cases("aligned", cases=TestAlignedSents)
def test_aligned_sents__words_attached(aligned):
    for pair in aligned.aligned_words:
        assert pair.src in pair.tgt.aligned
        assert pair.tgt in pair.src.aligned


@parametrize_with_cases("aligned", cases=TestAlignedSents)
def test_aligned_sents__one_dummy(aligned):
    assert len([w for w in aligned.src if isinstance(w, Null)]) == 1
    assert len([w for w in aligned.tgt if isinstance(w, Null)]) == 1


@parametrize_with_cases("aligned", cases=TestAlignedSents)
def test_aligned_sents__seq_spans_are_seq(aligned):
    # span pairs are tuple(src, tgt, is_mwe). So don't check mwe values for spantype
    # (they don't have spantypes because they are bools...)
    assert all(
        span.span_type == SpanType.SEQ
        for pair in aligned.aligned_seq_spans
        for span in pair
        if not isinstance(span, bool)
    )
