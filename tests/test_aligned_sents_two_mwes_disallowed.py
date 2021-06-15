from pytest_cases import parametrize_with_cases

from .conftest import TestAlignedSents


@parametrize_with_cases("aligned", cases=TestAlignedSents, glob="*two_mwgs_disallowed")
def test_aligned_sents__word_cross(aligned):
    # NULL-to-NULL alignments included
    assert len(aligned.aligned_words) == 19
    assert aligned.word_cross == 18


@parametrize_with_cases("aligned", cases=TestAlignedSents, glob="*two_mwgs_disallowed")
def test_aligned_sents__seq(aligned):
    # NULL-NULL is a separate group
    assert len(aligned.aligned_seq_spans) == 19
    assert aligned.seq_cross == 18
