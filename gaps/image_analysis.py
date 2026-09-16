from typing import Literal, Sequence

from gaps.fitness import EdgeOrientation, dissimilarity_measure
from gaps.piece import Piece
from gaps.progress_bar import print_progress

Orientation = Literal["T", "R", "D", "L"]

_EDGE_ORIENTATIONS: tuple[EdgeOrientation, ...] = ("LR", "TD")
_ORIENTATIONS: tuple[Orientation, ...] = ("T", "R", "D", "L")
_EDGE_TO_ORIENTATIONS: dict[EdgeOrientation, tuple[Orientation, Orientation]] = {
    "LR": ("L", "R"),
    "TD": ("T", "D"),
}


class ImageAnalysis:
    """Cache edge comparisons and best matches for one puzzle image."""

    def __init__(self) -> None:
        self.dissimilarity_measures: dict[
            tuple[int, int], dict[EdgeOrientation, float]
        ] = {}
        self.best_match_table: dict[int, dict[Orientation, list[tuple[int, float]]]] = (
            {}
        )

    def analyze_image(self, pieces: Sequence[Piece]) -> None:
        """Calculate and cache all pairwise edge comparisons."""
        self.dissimilarity_measures.clear()
        self.best_match_table.clear()

        for piece in pieces:
            self.best_match_table[piece.id] = {
                orientation: [] for orientation in _ORIENTATIONS
            }

        iterations = len(pieces) - 1
        if iterations <= 0:
            return

        for first_index in range(iterations):
            print_progress(first_index + 1, iterations, prefix="=== Analyzing image:")
            first_piece = pieces[first_index]
            for second_piece in pieces[first_index + 1 :]:
                for orientation in _EDGE_ORIENTATIONS:
                    self._update_best_match_table(
                        first_piece, second_piece, orientation
                    )
                    self._update_best_match_table(
                        second_piece, first_piece, orientation
                    )

        for piece_matches in self.best_match_table.values():
            for matches in piece_matches.values():
                matches.sort(key=lambda match: match[1])

    def _update_best_match_table(
        self,
        first_piece: Piece,
        second_piece: Piece,
        orientation: EdgeOrientation,
    ) -> None:
        measure = dissimilarity_measure(first_piece, second_piece, orientation)
        self.put_dissimilarity((first_piece.id, second_piece.id), orientation, measure)
        first_orientation, second_orientation = _EDGE_TO_ORIENTATIONS[orientation]
        self.best_match_table[second_piece.id][first_orientation].append(
            (first_piece.id, measure)
        )
        self.best_match_table[first_piece.id][second_orientation].append(
            (second_piece.id, measure)
        )

    def put_dissimilarity(
        self,
        ids: tuple[int, int],
        orientation: EdgeOrientation,
        value: float,
    ) -> None:
        self.dissimilarity_measures.setdefault(ids, {})[orientation] = value

    def get_dissimilarity(
        self, ids: tuple[int, int], orientation: EdgeOrientation
    ) -> float:
        return self.dissimilarity_measures[ids][orientation]

    def best_match(self, piece_id: int, orientation: Orientation) -> int:
        matches = self.best_match_table[piece_id][orientation]
        if not matches:
            raise ValueError(f"No best match available for piece {piece_id}")
        return matches[0][0]
