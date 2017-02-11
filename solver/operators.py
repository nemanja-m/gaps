import random
import bisect

def select(population):
    """Roulette wheel selection.

    Each individual is selected to reproduce, with probability directly
    proportional to its fitness score.

    :params population: Collection of the individuals for selecting.

    Usage::

        >>> from operators import select
        >>> population = [ 'create population' ]
        >>> parent = select(population)

    """

    fitnesses             = [individual.fitness for individual in population]
    probability_intervals = [sum(fitnesses[:i+1]) for i in range(len(fitnesses))]

    random_select  = random.uniform(0, probability_intervals[-1])
    selected_index = bisect.bisect_left(probability_intervals, random_select)

    return population[selected_index]


