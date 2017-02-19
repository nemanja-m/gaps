from solver import helpers
from solver import fitness
from solver.operators  import select
from solver.operators  import crossover
from solver.models import Individual
from solver.cache import DissimilarityMeasureCache
from operator import attrgetter
from pqdict import minpq

import time

def start_evolution(image, piece_size=128, population_size=500, generations=50, verbose=True):
    starting_time       = time.time()
    total_running_time  = 0
    time_per_generation = []

    # Create population
    pieces, rows, columns = helpers.flatten_image(image, piece_size, indexed=True)
    population = [Individual(pieces, rows, columns) for i in range(population_size)]

    print "[INFO] Created population in {:.2f} s".format(time.time() - starting_time)

    # Analyze image pieces' properties
    analyze_image(pieces)

    if verbose:
        print "[INFO] Starting evolution with {} generations ...".format(generations)
        print "\n|{:^12} | {:^12} | {:^10} | {:^12}|".format("Generation", "Duration", "Hit Rato", "Miss Ratio")
        print "+-------------+--------------+------------+-------------+"

    for generation in range(generations):
        new_population = []

        start = time.time()

        map(fitness.evaluate, population)

        # Elitism
        new_population.extend(best_individual(population, n=4))

        while len(new_population) <= population_size:
            first_parent  = select(population)
            second_parent = select(population)
            child         = crossover(first_parent, second_parent)

            new_population.append(child)

        population = new_population

        if verbose:
            print "|{:^12} | {:>10.2f} s | {:>8.2f} % | {:>10.2f} %|".format(generation,
                                                                       time.time() - start,
                                                                       DissimilarityMeasureCache.hit_ratio(),
                                                                       DissimilarityMeasureCache.miss_ratio())

        DissimilarityMeasureCache.reset_stats()

    if verbose:
        print "+-------------+--------------+------------+-------------+"
        print "\n[INFO] Total running time {:.2f} s".format(time.time() - starting_time)

    map(fitness.evaluate, population)

    return best_individual(population)[0]

def analyze_image(pieces):
    start = time.time()

    best_match_table = {}

    # Initialize table with best matches for each piece for each edge
    for piece in pieces:
        # For each edge we keep best matches as a min priority_queue
        # Edges with lower dissimilarity_measure have higer priority
        best_match_table[piece.id] = {
            "T": minpq(),
            "R": minpq(),
            "D": minpq(),
            "L": minpq()
        }

    def build_best_match_table(first_piece, second_piece, orientation):
        measure = fitness.dissimilarity_measure(first_piece, second_piece, orientation)

        best_match_table[first_piece.id][orientation[1]].additem(second_piece.id, measure)
        best_match_table[second_piece.id][orientation[0]].additem(first_piece.id, measure)

    # Calcualate dissimilarity measures and best matches for each piece
    for first in range(len(pieces) - 1):
        for second in range(first + 1, len(pieces)):
            for orientation in ["LR", "TD"]:
                build_best_match_table(pieces[first], pieces[second], orientation)
                build_best_match_table(pieces[second], pieces[first], orientation)

    DissimilarityMeasureCache.best_match_table = best_match_table
    DissimilarityMeasureCache.reset_stats()

    print "\n[INFO] Analyzed in {:.2f} s Total pieces: {}".format(time.time() - start, len(best_match_table))

def best_individual(population, n=1):
    """Returns the fittest individual from population"""
    return sorted(population, key=attrgetter("fitness"))[-n:]

