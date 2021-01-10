from pytest_cases import parametrize_with_cases
from astred.pairs import IdxPair

from .conftest import TestAlignedSents


@parametrize_with_cases("aligned", cases=TestAlignedSents, glob="*three_words_seq")
def test_aligned_sents__word_cross(aligned):
    # NULL-to-NULL alignments included
    assert len(aligned.aligned_words) == 5
    assert aligned.word_cross == 3


@parametrize_with_cases("aligned", cases=TestAlignedSents, glob="*three_words_seq")
def test_aligned_sents__seq(aligned):
    # NULL-NULL is a separate group
    assert len(aligned.aligned_seq_spans) == 3
    assert aligned.seq_aligns == [IdxPair(0, 0), IdxPair(1, 2), IdxPair(2, 1)]
    assert aligned.seq_cross == 1
