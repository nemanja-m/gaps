from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

Image: TypeAlias = NDArray[np.uint8]
CostLookup: TypeAlias = Callable[[tuple[int, int], "EdgeAxis"], float]


class Direction(StrEnum):
    """Directions used to locate neighboring puzzle pieces."""

    TOP = "top"
    RIGHT = "right"
    BOTTOM = "bottom"
    LEFT = "left"

    @property
    def opposite(self) -> "Direction":
        return {
            Direction.TOP: Direction.BOTTOM,
            Direction.RIGHT: Direction.LEFT,
            Direction.BOTTOM: Direction.TOP,
            Direction.LEFT: Direction.RIGHT,
        }[self]


class EdgeAxis(StrEnum):
    """The axis along which two pieces are adjacent."""

    HORIZONTAL = "horizontal"
    VERTICAL = "vertical"


@dataclass(frozen=True, slots=True)
class PuzzleLayout:
    """Dimensions shared by every arrangement of one puzzle."""

    rows: int
    columns: int
    piece_size: int

    def __post_init__(self) -> None:
        if self.rows <= 0 or self.columns <= 0 or self.piece_size <= 0:
            raise ValueError("layout dimensions must be positive")

    @property
    def piece_count(self) -> int:
        return self.rows * self.columns


@dataclass(slots=True)
class Piece:
    """A square image fragment with a stable identifier."""

    image: Image
    identifier: int

    def __post_init__(self) -> None:
        image = np.asarray(self.image)
        if image.ndim != 3:
            raise ValueError("piece image must have three dimensions")
        if image.shape[0] != image.shape[1]:
            raise ValueError("piece image must be square")
        if image.dtype != np.uint8:
            raise ValueError("piece image must use uint8 pixels")
        self.image = image.copy()

    def __getitem__(self, index: Any) -> Any:
        return self.image[index]

    @property
    def shape(self) -> tuple[int, ...]:
        return self.image.shape

    @property
    def size(self) -> int:
        return self.image.shape[0]


@dataclass(slots=True)
class Arrangement:
    """A candidate arrangement of every piece in a puzzle."""

    pieces: list[Piece]
    layout: PuzzleLayout
    _piece_mapping: dict[int, int] = field(init=False, repr=False)
    _cached_score: float | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.pieces = list(self.pieces)
        if len(self.pieces) != self.layout.piece_count:
            raise ValueError("piece count does not match the puzzle layout")
        identifiers = [piece.identifier for piece in self.pieces]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("piece identifiers must be unique")
        self._piece_mapping = {
            piece_id: index for index, piece_id in enumerate(identifiers)
        }

    @classmethod
    def random(
        cls,
        pieces: Sequence[Piece],
        layout: PuzzleLayout,
        rng,
    ) -> "Arrangement":
        shuffled_pieces = list(pieces)
        rng.shuffle(shuffled_pieces)
        return cls(shuffled_pieces, layout)

    def __getitem__(self, row: int) -> list[Piece]:
        start = row * self.layout.columns
        return self.pieces[start : start + self.layout.columns]

    def score(self, cost_lookup: CostLookup) -> float:
        """Calculate and cache the arrangement's compatibility score."""
        if self._cached_score is None:
            total_cost = 1 / 1000
            for row in range(self.layout.rows):
                for column in range(self.layout.columns - 1):
                    ids = (
                        self[row][column].identifier,
                        self[row][column + 1].identifier,
                    )
                    total_cost += cost_lookup(ids, EdgeAxis.HORIZONTAL)
            for row in range(self.layout.rows - 1):
                for column in range(self.layout.columns):
                    ids = (
                        self[row][column].identifier,
                        self[row + 1][column].identifier,
                    )
                    total_cost += cost_lookup(ids, EdgeAxis.VERTICAL)
            self._cached_score = 1000 / total_cost
        return self._cached_score

    def piece_by_id(self, identifier: int) -> Piece:
        return self.pieces[self._piece_mapping[identifier]]

    def edge(self, piece_id: int, direction: Direction) -> int | None:
        piece_index = self._piece_mapping[piece_id]
        columns = self.layout.columns

        if direction is Direction.TOP and piece_index >= columns:
            return self.pieces[piece_index - columns].identifier
        if direction is Direction.RIGHT and piece_index % columns < columns - 1:
            return self.pieces[piece_index + 1].identifier
        if (
            direction is Direction.BOTTOM
            and piece_index < (self.layout.rows - 1) * columns
        ):
            return self.pieces[piece_index + columns].identifier
        if direction is Direction.LEFT and piece_index % columns > 0:
            return self.pieces[piece_index - 1].identifier
        return None
