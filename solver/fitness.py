import numpy as np

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
            fitness_value += dissimilarity_measure(individual[i][j], individual[i][j + 1])

    # For each two adjancent pieces in columns
    for i in range(individual.rows - 1):
        for j in range(individual.columns):
            fitness_value += dissimilarity_measure(individual[i][j], individual[i + 1][j])

    individual.fitness = fitness_value

def dissimilarity_measure(first_piece, second_piece, position="LR"):
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
        >>> dissimilarity_measure(p1, p2, position="TD")

    """

    rows, columns    = first_piece.shape
    color_difference = None

    # | L | - | R |
    if position == "LR":
        color_difference = first_piece[0:rows, columns - 1, 0:3] - second_piece[0:rows, 0, 0:3]

    # | T |
    #   |
    # | D |
    if position == "TD":
        color_difference = first_piece[rows - 1, 0:columns, 0:3] - second_piece[0, 0:columns, 0:3]

    squared_color_difference = np.power(color_difference, 2)
    color_difference_per_row = np.sum(squared_color_difference, axis=1)
    total_difference         = np.sum(color_difference_per_row, axis=0)

    return np.sqrt(total_difference)
