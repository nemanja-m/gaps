import time

from solver import helpers
from solver import fitness
from solver.operators.select import select
from solver.operators.crossover import Crossover
from solver.models import Individual
from operator import attrgetter

def start_evolution(image, piece_size=28, population_size=200, generations=100, verbose=True):
    starting_time       = time.time()
    total_running_time  = 0
    time_per_generation = []
    crossover_times     = []

    # Create population
    pieces, rows, columns = helpers.flatten_image(image, piece_size, indexed=True)
    population = [Individual(pieces, rows, columns) for i in range(population_size)]

    print "\n[INFO] Created population in {:.3f} s".format(time.time() - starting_time)

    # Analyze image pieces' properties
    helpers.analyze_image(pieces)

    print "[INFO] Starting evolution with {} individuals and {} generations ...".format(population_size, generations)
    if verbose:
        print "\n|{:^12} | {:^12} | {:^10} | {:^9} | {:^9} |".format("Generation", "Duration", "Evaluation", "Selection", "Crossover")
        print "+-------------+--------------+------------+-----------+-----------+"

    for generation in range(generations):
        start = time.time()

        new_population = []

        et = time.time()
        map(fitness.evaluate, population)
        evaluation_time = time.time() - et

        # Elitism
        new_population.extend(best_individual(population, n=4))

        st = time.time()
        selected_parents = select(population, elite=4)
        selection_time = time.time() - st

        for first_parent, second_parent in selected_parents:
            ct = time.time()
            child = Crossover(first_parent, second_parent).child()
            crossover_times.append(time.time() - ct)
            new_population.append(child)

        population = new_population

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

    map(fitness.evaluate, population)

    return best_individual(population)[0]

def best_individual(population, n=1):
    """Returns the fittest individual from population"""
    return sorted(population, key=attrgetter("fitness"))[-n:]

