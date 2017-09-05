"""Selects fittest individuals from given population."""

import random
import bisect

def roulette_selection(population, fitnesses, elite=4):
    """Roulette wheel selection.

    Each individual is selected to reproduce, with probability directly
    proportional to its fitness score.

    :params population: Collection of the individuals for selecting.
    :params fitnesses: Fitness values for given population.
    :params elite: Number of elite individuals passed to next generation.

    Usage::

        >>> from selection import roulette_selection
        >>> selected_parents = roulette_selection(population, fitnesses, 10)

    """
    probability_intervals = [sum(fitnesses[:i + 1]) for i in range(len(fitnesses))]

    def select_individual():
        """Selects random individual from population based on fitess value"""
        random_select = random.uniform(0, probability_intervals[-1])
        selected_index = bisect.bisect_left(probability_intervals, random_select)
        return population[selected_index]

    selected = []
    for i in xrange(len(population) - elite):
        first, second = select_individual(), select_individual()
        selected.append((first, second))

    return selected
