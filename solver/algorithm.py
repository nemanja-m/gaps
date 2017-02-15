from solver import helpers
from solver import fitness
from solver.operators  import select
from solver.operators  import crossover
from solver.models import Individual
from solver.cache import DissimilarityMeasureCache
from operator import attrgetter
import time

def best_individual(population):
    """Returns the fittest individual from population"""
    return min(population, key=attrgetter("fitness"))

def start_evolution(image, piece_size=28, population_size=1000, generations=5, verbose=True):
    starting_time       = time.time()
    total_running_time  = 0
    time_per_generation = []

    # Create population
    pieces, rows, columns = helpers.flatten_image(image, piece_size, indexed=True)
    population = [Individual(pieces, rows, columns) for i in range(population_size)]

    if verbose:
        print "[INFO] Created population in {:.2f} s".format(time.time() - starting_time)
        print "[INFO] Starting evolution with {} generations ...".format(generations)
        print "\n|{:^12} | {:^12} | {:^10} | {:^12}|".format("Generation", "Duration", "Hit Rato", "Miss Ratio")
        print "+-------------+--------------+------------+-------------+"

    for generation in range(generations):
        new_population = []

        start = time.time()

        # Parallelize this !
        map(fitness.evaluate, population)

        if verbose:
            print "|{:^12} | {:>10.2f} s | {:>8} % | {:>10} %|".format(generation,
                                                                       time.time() - start,
                                                                       DissimilarityMeasureCache.hit_ratio(),
                                                                       DissimilarityMeasureCache.miss_ratio())

        DissimilarityMeasureCache.reset_stats()

        while len(new_population) <= population_size:
            first_parent  = select(population)
            second_parent = select(population)
            child         = crossover(first_parent, second_parent)

            new_population.append(child)

        population = new_population

    if verbose:
        print "+-------------+--------------+------------+-------------+"
        print "\n[INFO] Total running time {:.2f} s".format(time.time() - starting_time)

    return best_individual(population)

