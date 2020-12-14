from __future__ import annotations  # so that we can use the class in typing

import operator
from typing import List, Tuple, Union, NamedTuple, Iterator

# Doesnt make much sense because all methods are about a list of idxs rather than the namedtuple itself...
class AlignedIdxs(NamedTuple):
    src: int
    tgt: int

    @classmethod
    def from_str(cls, aligns: str, sort: bool = True) -> List[AlignedIdxs]:
        return cls.from_list([tuple(map(int, align.split("-"))) for align in aligns.split()], sort=sort)

    @classmethod
    def from_list(cls, aligns: Iterator[Tuple[int]], sort: bool = True) -> List[AlignedIdxs]:
        aligned = [cls(*align) for align in aligns]

        return cls.sort_idxs(aligned) if sort else aligned

    @staticmethod
    def sort_idxs(idxs: Iterator[AlignedIdxs]) -> List[AlignedIdxs]:
        return sorted(idxs, key=operator.attrgetter("src", "tgt"))

    @staticmethod
    def to_str(aligns: Union[List[Union[Tuple[int, int], AlignedIdxs]]]) -> str:
        """Convert list of alignments (tuple of src, tgt) to GIZA/Pharaoh string """
        return " ".join([f"{src}-{tgt}" for src, tgt in aligns])
