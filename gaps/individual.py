from collections.abc import Sequence
import random

import numpy as np

from gaps import utils
from gaps.image_analysis import ImageAnalysis, Orientation
from gaps.piece import Piece


class Individual:
    """One possible arrangement of all pieces in a puzzle."""

    FITNESS_FACTOR = 1000

    def __init__(
        self,
        pieces: Sequence[Piece],
        rows: int,
        columns: int,
        analysis: ImageAnalysis,
        shuffle: bool = True,
    ) -> None:
        if not pieces:
            raise ValueError("an individual must contain at least one piece")
        if rows <= 0 or columns <= 0:
            raise ValueError("rows and columns must be positive")
        if len(pieces) != rows * columns:
            raise ValueError("piece count does not match the requested dimensions")

        self.pieces = list(pieces)
        self.rows = rows
        self.columns = columns
        self.analysis = analysis
        self._fitness: float | None = None

        if shuffle:
            random.shuffle(self.pieces)

        self._piece_mapping = {
            piece.id: index for index, piece in enumerate(self.pieces)
        }

    def __getitem__(self, row: int) -> list[Piece]:
        start = row * self.columns
        return self.pieces[start : start + self.columns]

    @property
    def fitness(self) -> float:
        """Return the cached fitness score for this arrangement."""
        if self._fitness is None:
            fitness_value = 1 / self.FITNESS_FACTOR

            for row in range(self.rows):
                for column in range(self.columns - 1):
                    ids = (self[row][column].id, self[row][column + 1].id)
                    fitness_value += self.analysis.get_dissimilarity(ids, "LR")

            for row in range(self.rows - 1):
                for column in range(self.columns):
                    ids = (self[row][column].id, self[row + 1][column].id)
                    fitness_value += self.analysis.get_dissimilarity(ids, "TD")

            self._fitness = self.FITNESS_FACTOR / fitness_value

        return self._fitness

    def piece_size(self) -> int:
        """Return the side length of each square piece."""
        return self.pieces[0].size

    def piece_by_id(self, identifier: int) -> Piece:
        """Return a piece by its stable identifier."""
        return self.pieces[self._piece_mapping[identifier]]

    def to_image(self) -> np.ndarray:
        """Render this arrangement as an image."""
        return utils.assemble_image(self.pieces, self.rows, self.columns)

    def edge(self, piece_id: int, orientation: Orientation) -> int | None:
        """Return the identifier of a neighboring piece, if one exists."""
        piece_index = self._piece_mapping[piece_id]

        if orientation == "T" and piece_index >= self.columns:
            return self.pieces[piece_index - self.columns].id
        if orientation == "R" and piece_index % self.columns < self.columns - 1:
            return self.pieces[piece_index + 1].id
        if orientation == "D" and piece_index < (self.rows - 1) * self.columns:
            return self.pieces[piece_index + self.columns].id
        if orientation == "L" and piece_index % self.columns > 0:
            return self.pieces[piece_index - 1].id

        return None
