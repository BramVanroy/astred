from enum import Enum, unique


class ReprEnum(Enum):
    def __repr__(self):
        return self.value

    def __str__(self):
        return self.value


@unique
class Direction(ReprEnum):
    FORWARD = "forward"
    BACKWARD = "backward"
    NEUTRAL = "neutral"


@unique
class SpanType(ReprEnum):
    SEQ = "seq"
    SACR = "sacr"


@unique
class Side(ReprEnum):
    SRC = "src"
    TGT = "tgt"


@unique
class EditOperation(ReprEnum):
    MATCH = "match"
    RENAME = "rename"
    INSERTION = "insertion"
    DELETION = "deletion"
