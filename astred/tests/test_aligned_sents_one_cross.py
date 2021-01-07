from pytest_cases import parametrize_with_cases

from astred.pairs import IdxPair

from .conftest import TestAlignedSents


@parametrize_with_cases("aligned", cases=TestAlignedSents, glob="*one_cross")
def test_aligned_sents__word_cross(aligned):
    # NULL-to-NULL alignments included
    assert len(aligned.aligned_words) == 5
    assert aligned.word_cross == 1


@parametrize_with_cases("aligned", cases=TestAlignedSents, glob="*one_cross")
def test_aligned_sents__seq(aligned):
    # NULL-NULL is a separate group with idxs 0-0
    assert len(aligned.aligned_seq_spans) == 5
    assert aligned.seq_aligns == [
        IdxPair(0, 0),
        IdxPair(1, 1),
        IdxPair(2, 3),
        IdxPair(3, 2),
        IdxPair(4, 4),
    ]
    assert aligned.seq_cross == 1
