from typing import Dict, List, Tuple

import numpy as np

from .progress_bar import print_progress


def dissimilarity_measure(first_piece, second_piece, orientation="LR"):
    """Calculates color difference over all neighboring pixels over all color channels.

    The dissimilarity measure relies on the premise that adjacent jigsaw pieces
    in the original image tend to share similar colors along their abutting
    edges, i.e., the sum (over all neighboring pixels) of squared color
    differences (over all three color bands) should be minimal. Let pieces pi ,
    pj be represented in normalized L*a*b* space by corresponding W x W x 3
    matrices, where W is the height/width of each piece (in pixels).

    :params first_piece:  First input piece for calculation.
    :params second_piece: Second input piece for calculation.
    :params orientation:  How input pieces are oriented.

                          LR => 'Left - Right'
                          TD => 'Top - Down'

    Usage::

        >>> from gaps.fitness import dissimilarity_measure
        >>> from gaps.piece import Piece
        >>> p1, p2 = Piece(), Piece()
        >>> dissimilarity_measure(p1, p2, orientation="TD")

    """
    rows, columns, _ = first_piece.shape()
    color_difference = None

    # | L | - | R |
    if orientation == "LR":
        color_difference = (
            first_piece[:rows, columns - 1, :] - second_piece[:rows, 0, :]
        )

    # | T |
    #   |
    # | D |
    if orientation == "TD":
        color_difference = (
            first_piece[rows - 1, :columns, :] - second_piece[0, :columns, :]
        )

    squared_color_difference = np.power(color_difference / 255.0, 2)
    color_difference_per_row = np.sum(squared_color_difference, axis=1)
    total_difference = np.sum(color_difference_per_row, axis=0)

    value = np.sqrt(total_difference)

    return value


class ImageAnalysis(object):
    """Cache for dissimilarity measures of individuals

    Class have static lookup table where keys are Piece's id's.  For each pair
    puzzle pieces there is a map with values representing dissimilarity measure
    between them. Each next generation have greater chance to use cached value
    instead of calculating measure again.

    Attributes:
        dissimilarity_measures: Dictionary with cached dissimilarity measures for pieces
        best_match_table: Dictionary with best matching piece for each edge and piece

    """

    dissimilarity_measures: Dict[Tuple, Dict[str, float]] = {}
    best_match_table: Dict[int, Dict[str, List[Tuple[int, float]]]] = {}

    @classmethod
    def analyze_image(cls, pieces):
        for piece in pieces:
            # For each edge we keep best matches as a sorted list.
            # Edges with lower dissimilarity_measure have higher priority.
            cls.best_match_table[piece.id] = {"T": [], "R": [], "D": [], "L": []}

        def update_best_match_table(first_piece, second_piece):
            measure = dissimilarity_measure(first_piece, second_piece, orientation)
            cls.put_dissimilarity(
                (first_piece.id, second_piece.id), orientation, measure
            )
            cls.best_match_table[second_piece.id][orientation[0]].append(
                (first_piece.id, measure)
            )
            cls.best_match_table[first_piece.id][orientation[1]].append(
                (second_piece.id, measure)
            )

        # Calculate dissimilarity measures and best matches for each piece.
        iterations = len(pieces) - 1
        for first in range(iterations):
            print_progress(first, iterations - 1, prefix="=== Analyzing image:")
            for second in range(first + 1, len(pieces)):
                for orientation in ["LR", "TD"]:
                    update_best_match_table(pieces[first], pieces[second])
                    update_best_match_table(pieces[second], pieces[first])

        for piece in pieces:
            for orientation in ["T", "L", "R", "D"]:
                cls.best_match_table[piece.id][orientation].sort(key=lambda x: x[1])

    @classmethod
    def put_dissimilarity(cls, ids, orientation, value):
        """Puts a new value in lookup table for given pieces

        :params ids:         Identifiers of puzzle pieces
        :params orientation: Orientation of puzzle pieces. Possible values are:
                             'LR' => 'Left-Right'
                             'TD' => 'Top-Down'
        :params value:       Value of dissimilarity measure

        Usage::

            >>> from gaps.image_analysis import ImageAnalysis
            >>> ImageAnalysis.put_dissimilarity([1, 2], "TD", 42)
        """
        if ids not in cls.dissimilarity_measures:
            cls.dissimilarity_measures[ids] = {}
        cls.dissimilarity_measures[ids][orientation] = value

    @classmethod
    def get_dissimilarity(cls, ids, orientation):
        """Returns previously cached dissimilarity measure for input pieces

        :params ids:         Identifiers of puzzle pieces
        :params orientation: Orientation of puzzle pieces. Possible values are:
                             'LR' => 'Left-Right'
                             'TD' => 'Top-Down'

        Usage::

            >>> from gaps.image_analysis import ImageAnalysis
            >>> ImageAnalysis.get_dissimilarity([1, 2], "TD")

        """
        return cls.dissimilarity_measures[ids][orientation]

    @classmethod
    def best_match(cls, piece, orientation):
        """ "Returns best match piece for given piece and orientation"""
        return cls.best_match_table[piece][orientation][0][0]
