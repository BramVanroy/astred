from pytest_cases import parametrize_with_cases

from astred.pairs import IdxPair

from .conftest import TestAlignedSents


@parametrize_with_cases("aligned", cases=TestAlignedSents, glob="*two_nulls")
def test_aligned_sents__word_cross(aligned):
    # NULL-to-NULL alignments included
    assert len(aligned.aligned_words) == 6
    assert aligned.word_cross == 0


@parametrize_with_cases("aligned", cases=TestAlignedSents, glob="*two_nulls")
def test_aligned_sents__seq(aligned):
    # NULL-NULL is a separate group
    assert len(aligned.aligned_seq_spans) == 4
    assert aligned.seq_aligns == [
        IdxPair(0, 0),
        IdxPair(0, 1),
        IdxPair(1, 2),
        IdxPair(2, 0),
    ]
    assert aligned.seq_cross == 0
