from pytest_cases import parametrize_with_cases

from .conftest import TestAlignedSents


@parametrize_with_cases("aligned", cases=TestAlignedSents, glob="*m_to_n")
def test_aligned_sents__aligned_cross(aligned):
    # NULL-to-NULL alignments included
    assert len(aligned.aligned_words) == 10
    assert aligned.word_cross == 3


@parametrize_with_cases("aligned", cases=TestAlignedSents, glob="*m_to_n")
def test_aligned_sents__seq(aligned):
    # NULL-NULL is a separate group
    assert len(aligned.aligned_seq_spans) == 10
    assert aligned.seq_cross == 3


@parametrize_with_cases("aligned", cases=TestAlignedSents, glob="*m_to_n")
def test_aligned_sents__words_cross(aligned):
    # NULL-to-NULL alignments included
    assert aligned.src[0].cross == 0
    assert aligned.src[1].cross == 0
    assert aligned.src[2].cross == 1
    assert aligned.src[3].cross == 1
    assert aligned.src[4].cross == 2
    assert aligned.src[5].cross == 2
    assert aligned.src[6].cross == 0

    assert aligned.tgt[0].cross == 0
    assert aligned.tgt[1].cross == 0
    assert aligned.tgt[2].cross == 1
    assert aligned.tgt[3].cross == 1
    assert aligned.tgt[4].cross == 2
    assert aligned.tgt[5].cross == 1
    assert aligned.tgt[6].cross == 1


@parametrize_with_cases("aligned", cases=TestAlignedSents, glob="*m_to_n")
def test_aligned_sents__words_avg_cross(aligned):
    # NULL-to-NULL alignments included
    assert aligned.src[0].avg_cross == 0.0
    assert aligned.src[1].avg_cross == 0.0
    assert aligned.src[2].avg_cross == 1 / 2
    assert aligned.src[3].avg_cross == 1.0
    assert aligned.src[4].avg_cross == 2 / 3
    assert aligned.src[5].avg_cross == 2.0
    assert aligned.src[6].avg_cross == 0.0

    assert aligned.tgt[0].avg_cross == 0.0
    assert aligned.tgt[1].avg_cross == 0.0
    assert aligned.tgt[2].avg_cross == 1 / 2
    assert aligned.tgt[3].avg_cross == 1.0
    assert aligned.tgt[4].avg_cross == 2 / 2
    assert aligned.tgt[5].avg_cross == 1.0
    assert aligned.tgt[6].avg_cross == 1.0
