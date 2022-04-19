"""Selects fittest individuals from given population."""

import random
import bisect

try:
    xrange  # Python 2
except NameError:
    xrange = range  # Python 3


def roulette_selection(population, elites=4):
    """Roulette wheel selection.

    Each individual is selected to reproduce, with probability directly
    proportional to its fitness score.

    :params population: Collection of the individuals for selecting.
    :params elite: Number of elite individuals passed to next generation.

    Usage::

        >>> from gaps.selection import roulette_selection
        >>> selected_parents = roulette_selection(population, 10)

    """
    fitness_values = [individual.fitness for individual in population]
    probability_intervals = [
        sum(fitness_values[: i + 1]) for i in range(len(fitness_values))
    ]

    def select_individual():
        """Selects random individual from population based on fitess value"""
        random_select = random.uniform(0, probability_intervals[-1])
        selected_index = bisect.bisect_left(probability_intervals, random_select)
        return population[selected_index]

    selected = []
    for i in xrange(len(population) - elites):
        first, second = select_individual(), select_individual()
        selected.append((first, second))

    return selected
