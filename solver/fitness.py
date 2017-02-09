from .helpers import dissimilarity_measure

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

    return fitness_value
