import time
import matplotlib.pyplot as plt

from solver import helpers
from solver import fitness
from solver.operators.select import select
from solver.operators.crossover import Crossover
from solver.models import Individual
from operator import attrgetter

class Algorithm:

    def __init__(self, image, piece_size, population_size, generations, verbose):
        self._image = image
        self._piece_size = piece_size
        self._population_size = population_size
        self._generations = generations
        self._verbose = verbose

        self._initialize_figure()

    def start_evolution(self):
        starting_time       = time.time()
        total_running_time  = 0
        time_per_generation = []
        crossover_times     = []

        # Create population
        pieces, rows, columns = helpers.flatten_image(self._image, self._piece_size, indexed=True)
        population = [Individual(pieces, rows, columns) for i in range(self._population_size)]

        print "\n[INFO] Created population in {:.3f} s".format(time.time() - starting_time)

        # Analyze image pieces' properties
        helpers.analyze_image(pieces)

        print "[INFO] Starting evolution with {} individuals and {} generations ...".format(self._population_size, self._generations)

        if self._verbose:
            print "\n|{:^12} | {:^12} | {:^10} | {:^9} | {:^9} |".format("Generation", "Duration", "Evaluation", "Selection", "Crossover")
            print "+-------------+--------------+------------+-----------+-----------+"

        for generation in range(self._generations):
            start = time.time()

            new_population = []

            et = time.time()
            map(fitness.evaluate, population)
            evaluation_time = time.time() - et

            # Elitism
            new_population.extend(self._best_individual(population, n=4))

            st = time.time()
            selected_parents = select(population, elite=4)
            selection_time = time.time() - st

            for first_parent, second_parent in selected_parents:
                ct = time.time()
                child = Crossover(first_parent, second_parent).child()
                crossover_times.append(time.time() - ct)
                new_population.append(child)

            population = new_population

            fittest = self._best_individual(population)[0]

            self._show_fittest(fittest.to_image(), title="Generation: {} / {}".format(generation + 1, self._generations))

            if self._verbose:
                print "|{:^12} | {:>10.3f} s | {:>8.3f} s | {:>7.3f} s | {:>7.3f} s |".format(generation + 1,
                                                                                              time.time() - start,
                                                                                              evaluation_time,
                                                                                              selection_time,
                                                                                              sum(crossover_times))

            crossover_times = []

        if self._verbose:
            print "+-------------+--------------+------------+-----------+-----------+"
            print "\n[INFO] Total running time {:.2f} s\n".format(time.time() - starting_time)

        map(fitness.evaluate, population)

        return self._best_individual(population)[0]

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

        self._current_image = ax.imshow(self._image, aspect="auto", animated=True, interpolation="bilinear")
        self._show_fittest(self._image, "Initial problem")
