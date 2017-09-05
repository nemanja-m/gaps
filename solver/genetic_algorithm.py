import time
from operator import attrgetter

from solver import image_helpers
from solver.selection import roulette_selection
from solver.crossover import Crossover
from solver.individual import Individual
from solver.image_analysis import ImageAnalysis
from solver.plot import Plot


class GeneticAlgorithm(object):
    ELITISM_FACTOR = 0.02

    def __init__(self, image, piece_size, population_size, generations):
        self._image = image
        self._piece_size = piece_size
        self._generations = generations
        self._elite_size = int(population_size * self.ELITISM_FACTOR)

        pieces, rows, columns = image_helpers.flatten_image(image, piece_size, indexed=True)
        self._population = [Individual(pieces, rows, columns) for _ in range(population_size)]

        ImageAnalysis.analyze_image(pieces)

    def start_evolution(self, verbose):
        plot = Plot(self._image)
        fittest = None

        for generation in range(self._generations):
            new_population = []

            # Elitism
            elite = self._get_elite_individuals(elites=self._elite_size)
            new_population.extend(elite)

            selected_parents = roulette_selection(self._population, elites=self._elite_size)

            for first_parent, second_parent in selected_parents:
                crossover = Crossover(first_parent, second_parent)
                crossover.run()
                child = crossover.child()
                new_population.append(child)

            self._population = new_population
            fittest = self._best_individual()
            plot.show_fittest(fittest.to_image(), title="Generation: {} / {}".format(generation + 1, self._generations))

        return fittest

    def _get_elite_individuals(self, elites):
        """Returns first 'elite_count' fittest individuals from population"""
        return sorted(self._population, key=attrgetter("fitness"))[-elites:]

    def _best_individual(self):
        """Returns the fittest individual from population"""
        return max(self._population, key=attrgetter("fitness"))
