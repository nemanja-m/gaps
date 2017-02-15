import numpy as np
from solver.models import Piece

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
            left, top, w, h = y * piece_size, x * piece_size, (y + 1) * piece_size, (x + 1) * piece_size

            piece = np.empty((piece_size, piece_size, 3))
            piece[0:piece_size, 0:piece_size, :] = image[left:w, top:h, :]

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

    return np.vstack(vertical_stack).astype(np.uint8)

