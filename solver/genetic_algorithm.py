from operator import attrgetter
import time
import matplotlib.pyplot as plt

from solver import image_helpers
from solver.selection import roulette_selection
from solver.crossover import Crossover
from solver.individual import Individual
from solver.image_analysis import ImageAnalysis


class GeneticAlgorithm(object):
    ELITISM_FACTOR = 0.02

    def __init__(self, image, piece_size):
        self._image = image
        self._piece_size = piece_size

        self._initialize_figure()

    def start_evolution(self, population_size, generations, verbose):
        starting_time = time.time()
        crossover_times = []

        elite_size = int(population_size * self.ELITISM_FACTOR)

        # Create population
        pieces, rows, columns = image_helpers.flatten_image(self._image, self._piece_size, indexed=True)
        population = [Individual(pieces, rows, columns) for i in range(population_size)]

        print "\n[INFO] Created population in {:.3f} s".format(time.time() - starting_time)

        ImageAnalysis.analyze_image(pieces)

        print "[INFO] Starting evolution with {} individuals and {} generations ...".format(population_size, generations)

        if verbose:
            print "\n|{:^12} | {:^12} | {:^10} | {:^9} | {:^9} |".format("Generation", "Duration", "Evaluation", "Selection", "Crossover")
            print "+-------------+--------------+------------+-----------+-----------+"

        fittest = None

        for generation in range(generations):
            start = time.time()

            new_population = []

            et = time.time()

            evaluation_time = time.time() - et

            # Elitism
            new_population.extend(self._best_individual(population, n=elite_size))

            st = time.time()
            selected_parents = roulette_selection(population, elite=elite_size)
            selection_time = time.time() - st

            for first_parent, second_parent in selected_parents:
                ct = time.time()
                child = Crossover(first_parent, second_parent).start().child()
                new_population.append(child)
                crossover_times.append(time.time() - ct)

            population = new_population

            fittest = self._best_individual(population)[0]

            self._show_fittest(fittest.to_image(), title="Generation: {} / {}".format(generation + 1, generations))

            if verbose:
                print "|{:^12} | {:>10.3f} s | {:>8.3f} s | {:>7.3f} s | {:>7.3f} s |".format(generation + 1,
                                                                                              time.time() - start,
                                                                                              evaluation_time,
                                                                                              selection_time,
                                                                                              sum(crossover_times))

            crossover_times = []

        if verbose:
            print "+-------------+--------------+------------+-----------+-----------+"
            print "\n[INFO] Total running time {:.2f} s\n".format(time.time() - starting_time)

        return fittest

    def _best_individual(self, population, n=1):
        """Returns the fittest individual from population"""
        return sorted(population, key=attrgetter("fitness"))[-n:]

    def _show_fittest(self, fittest, title):
        plt.suptitle(title, fontsize=20)
        self._current_image.set_data(fittest)
        plt.draw()

        # Give pyplot .1 s to draw image
        plt.pause(0.1)

    def _initialize_figure(self):
        # Let image fill the figure
        fig = plt.figure(frameon=False)
        ax = plt.Axes(fig, [0., 0., 1., .9])
        ax.set_axis_off()
        fig.add_axes(ax)

        self._current_image = ax.imshow(self._image, aspect="auto", animated=True)
        self._show_fittest(self._image, "Initial problem")
