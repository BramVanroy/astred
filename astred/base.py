from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from statistics import mean
from typing import Any, Dict, List

from .enum import Direction, Side


@dataclass(eq=False)
class Crossable:
    id: int
    doc: Any = field(default=None, repr=False)

    aligned: List[Crossable] = field(default_factory=list, init=False, repr=False)
    aligned_directions: Dict[int, Direction] = field(default_factory=dict, init=False, repr=False)
    aligned_cross: Dict[int, int] = field(default_factory=dict, init=False, repr=False)

    is_null: bool = field(default=False)

    def __post_init__(self):
        if self.id < 0:
            raise ValueError("id must be a positive integer (or 0 for null items)")
        elif self.id == 0 and not self.is_null:
            raise ValueError("id cannot be 0 for non-null items")

        if self.__class__ == Crossable:
            raise TypeError(f"Cannot instantiate abstract class {self.__class__.__name__}.")

    def __repr__(self):
        return f"{self.__class__.__name__}(id={self.id}, side={self.side}, text={self.text})"

    @property
    def side(self) -> Side:
        return self.doc.side if self.doc else None

    @property
    def avg_cross(self) -> float:
        return mean(self.aligned_cross.values()) if self.aligned_cross else 0

    @property
    def cross(self) -> int:
        return sum(self.aligned_cross.values())

    def add_aligned(self, item):
        self.aligned.append(item)
        self.aligned_directions[item.id] = (
            Direction.NEUTRAL if item.id == self.id else Direction.FORWARD if item.id > self.id else Direction.BACKWARD
        )
        self.aligned_cross[item.id] = 0

    def get_direction_to_item(self, item):
        idx = self.aligned.index(item)
        return self.aligned_directions[idx]


class SpanMixin(ABC):
    @property
    def text(self):
        return " ".join([w.text for w in self.no_null_words])

    @property
    def word_idxs(self):
        return [w.id for w in self.words]

    def __getitem__(self, idx):
        return self.words[idx]

    def __iter__(self):
        return iter(self.words)

    def __len__(self):
        return len(self.words)

    @property
    def num_changes(self, attr="deprel"):
        return sum([w.num_changes(attr=attr) for w in self])

    @property
    def no_null_words(self):
        return [w for w in self.words if not w.is_null]

    @abstractmethod
    def attach_self_to_words(self):
        raise NotImplementedError()
