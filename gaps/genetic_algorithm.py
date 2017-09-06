from operator import attrgetter
from gaps import image_helpers
from gaps.selection import roulette_selection
from gaps.crossover import Crossover
from gaps.individual import Individual
from gaps.image_analysis import ImageAnalysis
from gaps.plot import Plot
from gaps.progress_bar import print_progress


class GeneticAlgorithm(object):
    ELITISM_FACTOR = 0.02

    def __init__(self, image, piece_size, population_size, generations):
        self._image = image
        self._piece_size = piece_size
        self._generations = generations
        self._elite_size = int(population_size * self.ELITISM_FACTOR)
        pieces, rows, columns = image_helpers.flatten_image(image, piece_size, indexed=True)
        self._population = [Individual(pieces, rows, columns) for _ in range(population_size)]
        self._pieces = pieces

    def start_evolution(self, verbose):
        if verbose:
            plot = Plot(self._image)

        ImageAnalysis.analyze_image(self._pieces)
        fittest = None

        for generation in range(self._generations):
            print_progress(generation, self._generations - 1, prefix="Solving puzzle:")

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

            fittest = self._best_individual()
            self._population = new_population

            if verbose:
                plot.show_fittest(fittest.to_image(), "Generation: {} / {}".format(generation + 1, self._generations))

        return fittest

    def _get_elite_individuals(self, elites):
        """Returns first 'elite_count' fittest individuals from population"""
        return sorted(self._population, key=attrgetter("fitness"))[-elites:]

    def _best_individual(self):
        """Returns the fittest individual from population"""
        return max(self._population, key=attrgetter("fitness"))
