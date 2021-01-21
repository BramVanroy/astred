from pytest_cases import parametrize_with_cases

from astred.pairs import IdxPair

from .conftest import TestAlignedSents


@parametrize_with_cases("aligned", cases=TestAlignedSents, glob="*long_distance_cross")
def test_aligned_sents__aligned_cross(aligned):
    # NULL-to-NULL alignments included
    assert len(aligned.aligned_words) == 5
    assert aligned.word_cross == 5


@parametrize_with_cases("aligned", cases=TestAlignedSents, glob="*long_distance_cross")
def test_aligned_sents__words_cross(aligned):
    # NULL-to-NULL alignments included
    assert aligned.src[0].cross == 0
    assert aligned.src[1].cross == 3
    assert aligned.src[2].cross == 2
    assert aligned.src[3].cross == 2
    assert aligned.src[4].cross == 3

    assert aligned.tgt[0].cross == 0
    assert aligned.tgt[1].cross == 3
    assert aligned.tgt[2].cross == 2
    assert aligned.tgt[3].cross == 2
    assert aligned.tgt[4].cross == 3


@parametrize_with_cases("aligned", cases=TestAlignedSents, glob="*long_distance_cross")
def test_aligned_sents__words_avg_cross(aligned):
    # all these are the same as regular cross because there are no m-to-n alignments
    # avg_cross is a float, though
    # NULL-to-NULL alignments included
    assert aligned.src[0].avg_cross == 0.0
    assert aligned.src[1].avg_cross == 3.0
    assert aligned.src[2].avg_cross == 2.0
    assert aligned.src[3].avg_cross == 2.0
    assert aligned.src[4].avg_cross == 3.0

    assert aligned.tgt[0].avg_cross == 0.0
    assert aligned.tgt[1].avg_cross == 3.0
    assert aligned.tgt[2].avg_cross == 2.0
    assert aligned.tgt[3].avg_cross == 2.0
    assert aligned.tgt[4].avg_cross == 3.0


@parametrize_with_cases("aligned", cases=TestAlignedSents, glob="*long_distance_cross")
def test_aligned_sents__seq(aligned):
    # NULL-NULL is a separate group
    assert len(aligned.aligned_seq_spans) == 4
    assert aligned.seq_aligns == [
        IdxPair(0, 0),
        IdxPair(1, 3),
        IdxPair(2, 2),
        IdxPair(3, 1),
    ]
    assert aligned.seq_cross == 3
