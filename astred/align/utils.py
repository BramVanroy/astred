from typing import Generator, List, NamedTuple, Tuple, Union

AlignedIdxs = NamedTuple("AlignedIdxs", [("src", int), ("tgt", int)])


def aligns_from_str(aligns: str) -> List:
    return sorted([AlignedIdxs(*map(int, align.split("-"))) for align in aligns.split()])


def aligns_to_str(aligns: Union[List[Union[Tuple[int, int], AlignedIdxs]]]) -> str:
    """Convert list of alignments (tuple of src, tgt) to GIZA/Pharaoh string """
    return " ".join([f"{src}-{tgt}" for src, tgt in aligns])
