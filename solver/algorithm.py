from solver import helpers
from solver import fitness
from solver.operators  import select
from solver.operators  import crossover
from solver.models import Individual
from solver.cache import DissimilarityMeasureCache
from operator import attrgetter
from pqdict import minpq

import time
import numpy as np

def start_evolution(image, piece_size=28, population_size=300, generations=30, verbose=True):
    starting_time       = time.time()
    total_running_time  = 0
    time_per_generation = []
    crossover_times     = []

    # Create population
    pieces, rows, columns = helpers.flatten_image(image, piece_size, indexed=True)
    population = [Individual(pieces, rows, columns) for i in range(population_size)]

    print "[INFO] Created population in {:.2f} s".format(time.time() - starting_time)

    # Analyze image pieces' properties
    analyze_image(pieces)

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
            child = crossover(first_parent, second_parent)
            crossover_times.append(time.time() - ct)
            new_population.append(child)

        population = new_population

        if verbose:
            print "|{:^12} | {:>10.3f} s | {:>8.3f} s | {:>7.3f} s | {:>7.3f} s |".format(generation,
                                                                       time.time() - start,
                                                                       evaluation_time,
                                                                       selection_time,
                                                                       sum(crossover_times))

        crossover_times = []

    if verbose:
        print "+-------------+--------------+------------+-----------+-----------+"
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

