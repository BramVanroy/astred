from pytest_cases import parametrize_with_cases

from .conftest import TestAlignedSents


@parametrize_with_cases("aligned", cases=TestAlignedSents, glob="*mwe_and_cross")
def test_aligned_sents__word_cross(aligned):
    # NULL-to-NULL alignments included
    assert len(aligned.aligned_words) == 19
    assert aligned.word_cross == 69


@parametrize_with_cases("aligned", cases=TestAlignedSents, glob="*mwe_and_cross")
def test_aligned_sents__seq(aligned):
    # NULL-NULL is a separate group
    assert len(aligned.aligned_seq_spans) == 4
    assert aligned.seq_cross == 3
