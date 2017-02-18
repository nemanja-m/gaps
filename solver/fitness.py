import numpy as np
from solver.cache import DissimilarityMeasureCache

def evaluate(individual):
    """Evaluates fitness value for given individual.

    Fitness value is calculated as sum of dissimilarity measures between each adjacent pieces.

    :params individual: One sample from population which fitness value is evaluated.

    Usage::

        >>> from fitness import evaluate
        >>> ind = toolbox.Individual()
        >>> ind.fitness.value = evaluate(ind)

    """

    fitness_value = 0

    # For each two adjancent pieces in rows
    for i in range(individual.rows):
        for j in range(individual.columns - 1):
            fitness_value += dissimilarity_measure(individual[i][j], individual[i][j + 1], orientation="LR")

    # For each two adjancent pieces in columns
    for i in range(individual.rows - 1):
        for j in range(individual.columns):
            fitness_value += dissimilarity_measure(individual[i][j], individual[i + 1][j], orientation="TD")

    individual.fitness = 1000 / fitness_value

def dissimilarity_measure(first_piece, second_piece, orientation="LR"):
    """Calculates color difference over all neighboring pixels over all color channels.

    The dissimilarity measure relies on the premise that adjacent jigsaw pieces in the original image tend to share
    similar colors along their abutting edges, i.e., the sum (over all neighboring pixels) of squared color differences (over all
    three color bands) should be minimal. Let pieces pi , pj be represented in normalized L*a*b* space by corresponding
    W x W x 3 matrices, where W is the height/width of each piece (in pixels).

    :params first_piece:  First input piece for calculation.
    :params second_piece: Second input piece for calculation.
    :params position:     How input pieces are positioned.

                          LR => 'Left - Right'
                          TD => 'Top - Down'

    Usage::

        >>> from fitnes import dissimilarity_measure
        >>> dissimilarity_measure(p1, p2, orientation="TD")

    """

    ids = [first_piece.id, second_piece.id]

    # Return cached value if it exists
    if DissimilarityMeasureCache.contains(ids, orientation):
        return DissimilarityMeasureCache.get(ids, orientation)

    rows, columns, _ = first_piece.shape()
    color_difference = None

    # | L | - | R |
    if orientation == "LR":
        color_difference = first_piece[:rows, columns - 1, :] - second_piece[:rows, 0, :]

    # | T |
    #   |
    # | D |
    if orientation == "TD":
        color_difference = first_piece[rows - 1, :columns, :] - second_piece[0, :columns, :]

    squared_color_difference = np.power(color_difference / 255.0, 2)
    color_difference_per_row = np.sum(squared_color_difference, axis=1)
    total_difference         = np.sum(color_difference_per_row, axis=0)

    value = np.sqrt(total_difference)

    # Cache calculated dissimilarity measure
    DissimilarityMeasureCache.put(ids, orientation, value)

    return value

