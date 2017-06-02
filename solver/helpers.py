import numpy as np
import time

from solver.models import Piece
from solver import fitness
from cache import Cache
from sortedcontainers import SortedListWithKey

def flatten_image(image, piece_size, indexed=False):
    """Converts image into list of square pieces.

    Input image is divided into square pieces of specified size and than
    flattened into list. Eacch list element is PIECE_SIZE x PIECE_SIZE x 3

    :params image:      Input image.
    :params piece_size: Size of single square piece. Each piece is PIECE_SIZE x PIECE_SIZE
    :params indexed:    If True list of Pieces with IDs will be returned, otherwise just plain list of ndarray pieces

    Usage::

        >>> from helpers import flatten_image
        >>> flat_image = flatten_image(image, 32)

    """

    rows, columns = image.shape[0] // piece_size, image.shape[1] // piece_size

    pieces = []

    # Crop pieces from original image
    for y in range(rows):
        for x in range(columns):
            left, top, w, h =  x * piece_size, y * piece_size, (x + 1) * piece_size, (y + 1) * piece_size

            piece = np.empty((piece_size, piece_size, image.shape[2]))
            piece[:piece_size, :piece_size, :] = image[top:h, left:w, :]

            pieces.append(piece)

    if indexed == True:
        pieces = [Piece(value, index) for index, value in enumerate(pieces)]

    return pieces, rows, columns

def assemble_image(pieces, rows, columns):
    """Assembles image from pieces.

    Given an array of pieces and desired image dimensions, function
    assembles image by stacking pieces.

    :params pieces:  Image pieces as an array.
    :params rows:    Number of rows in resulting image.
    :params columns: Number of columns in resulting image.

    Usage::

        >>> from helpers import assemble_image
        >>> from helpers import flatten_image
        >>> pieces, rows, cols = flatten_image(image, 32)
        >>> original_img = assemble_image(pieces, rows, cols)

    """

    vertical_stack = []

    for i in range(rows):
        horizontal_stack = []
        for j in range(columns):
            horizontal_stack.append(pieces[i * columns + j])

        vertical_stack.append(np.hstack(horizontal_stack))

    return np.vstack(vertical_stack) / 255.0

def analyze_image(pieces):
    start = time.time()

    best_match_table = {}

    # Initialize table with best matches for each piece for each edge
    for piece in pieces:
        # For each edge we keep best matches as a min priority_queue
        # Edges with lower dissimilarity_measure have higer priority
        best_match_table[piece.id] = {
            "T": SortedListWithKey(key=lambda k: k[1]),
            "R": SortedListWithKey(key=lambda k: k[1]),
            "D": SortedListWithKey(key=lambda k: k[1]),
            "L": SortedListWithKey(key=lambda k: k[1])
        }

    def build_best_match_table(first_piece, second_piece, orientation):
        measure = fitness.dissimilarity_measure(first_piece, second_piece, orientation)

        best_match_table[second_piece.id][orientation[0]].add((first_piece.id, measure))
        best_match_table[first_piece.id][orientation[1]].add((second_piece.id, measure))

    # Calcualate dissimilarity measures and best matches for each piece
    for first in range(len(pieces) - 1):
        for second in range(first + 1, len(pieces)):
            for orientation in ["LR", "TD"]:
                build_best_match_table(pieces[first], pieces[second], orientation)
                build_best_match_table(pieces[second], pieces[first], orientation)

    Cache.best_match_table = best_match_table

    print "[INFO] Analyzed in {:.3f} s Total pieces: {}".format(time.time() - start, len(best_match_table))

