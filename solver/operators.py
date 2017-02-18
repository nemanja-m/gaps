import random
import bisect
import sys

from pqdict import minpq
from solver.models import Individual

def select(population):
    """Roulette wheel selection.

    Each individual is selected to reproduce, with probability directly
    proportional to its fitness score.

    :params population: Collection of the individuals for selecting.

    Usage::

        >>> from operators import select
        >>> population = [ 'create population' ]
        >>> parent = select(population)

    """

    fitnesses             = [individual.fitness for individual in population]
    probability_intervals = [sum(fitnesses[:i + 1]) for i in range(len(fitnesses))]

    random_select  = random.uniform(0, probability_intervals[-1])
    selected_index = bisect.bisect_left(probability_intervals, random_select)

    return population[selected_index]

def crossover(first_parent, second_parent, debug=False):
    child_pieces    = {}
    taken_positions = set()
    recommended_taken = set()
    unreached_pieces = []

    # Initialize recommended positions with root piece
    root = first_parent.pieces[int(random.uniform(0, len(first_parent.pieces)))]
    recommended_positions = { root.id: (0, 0) }
    recommended_taken.add((0, 0))

    min_row  = 0
    max_row  = 0
    min_column  = 0
    max_column  = 0

    # Initialize priority queue
    priority_queue = minpq()

    for index in range(len(first_parent.pieces)):
        priority_queue[index] = float("inf")

    priority_queue[root.id] = 0

    for selected, value in priority_queue.popitems():
        if value == float('inf'):
            unreached_pieces.append(selected)
            continue

        row, column = recommended_positions[selected]

        child_pieces[selected] = (row, column)
        taken_positions.add((row, column))

        edges = unvisited_edges(selected, first_parent, second_parent, visited=child_pieces)

        for src, dst, orientation, weight in edges:
            relative_position = child_pieces[selected]

            offset = 0

            if orientation == "T":
                offset = (-1, 0)
            elif orientation == "R":
                offset = (0, 1)
            elif orientation == "D":
                offset = (1, 0)
            else:
                offset = (0, -1)

            row, column = tuple(map(sum, zip(relative_position, offset)))

            if (row, column) in taken_positions:
                continue

            if (row, column) in recommended_taken:
                continue

            if abs(min(min_row, row)) + abs(max(max_row, row)) >= first_parent.rows:
                continue

            if abs(min(min_column, column)) + abs(max(max_column, column)) >= first_parent.columns:
                continue

            min_row = min(min_row, row)
            max_row = max(max_row, row)
            min_column = min(min_column, column)
            max_column = max(max_column, column)

            if shared_edge(src, dst, orientation, first_parent, second_parent):
                priority_queue[dst] = -float("inf")
                recommended_positions[dst] = (row, column)
                recommended_taken.add((row, column))

            elif best_buddy_edge(src, dst, first_parent, second_parent):
                priority_queue[dst] = -sys.maxint
                recommended_positions[dst] = (row, column)
                recommended_taken.add((row, column))

            elif dst in priority_queue and priority_queue[dst] > weight:
                priority_queue[dst] = weight
                recommended_positions[dst] = (row, column)
                recommended_taken.add((row, column))

    pieces = [None] * (first_parent.rows * first_parent.columns)

    for piece, (row, column) in child_pieces.iteritems():
        index = (row - min_row) * first_parent.columns + (column - min_column)
        pieces[index] = first_parent.piece_by_id(piece)

    if len(unreached_pieces) > 0:
        for idx, element in enumerate(pieces):
            if element == None:
                piece_id = unreached_pieces.pop()
                pieces[idx] = first_parent.piece_by_id(piece_id)

    return Individual(pieces, first_parent.rows, first_parent.columns, shuffle=False)

def shared_edge(src, dst, orientation, first_parent, second_parent):
    return (first_parent.contains_edge(src, dst, orientation) and
            second_parent.contains_edge(src, dst, orientation))

def best_buddy_edge(src, dst, first_parent, second_parent):
    return first_parent.best_buddies(src, dst) or second_parent.best_buddies(src, dst)

def unvisited_edges(piece, first_parent, second_parent, visited):
    """Returns all unvisited edges for a given piece from both parents"""
    fp_edges = [edge for edge in first_parent.edges(piece) if not edge[1]in visited]
    sp_edges = [edge for edge in second_parent.edges(piece) if not edge[1]in visited]

    return fp_edges + sp_edges

