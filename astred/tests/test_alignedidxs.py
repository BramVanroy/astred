import pytest

from astred.align.utils import AlignedIdxs


@pytest.fixture
def idx_tuples():
    return [(0, 0), (1, 1), (2, 0), (3, 2), (3, 3), (4, 4)]

@pytest.fixture
def idx_str():
    return "0-0 1-1 2-0 3-2 3-3 4-4"

@pytest.fixture
def aligned_idxs(idx_tuples):
    # Ground truth. These must be correct because we simply create
    # named tuples out of... tuples
    return [AlignedIdxs(src, tgt) for src, tgt in idx_tuples]


def test_from_list(aligned_idxs, idx_tuples):
    assert aligned_idxs == AlignedIdxs.from_list(idx_tuples)


def test_from_str(aligned_idxs, idx_str):
    assert aligned_idxs == AlignedIdxs.from_str(idx_str)
