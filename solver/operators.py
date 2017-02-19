import random
import bisect

from solver.cache import DissimilarityMeasureCache
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

# Quick hack !!!
min_row  = 0
max_row  = 0
min_column  = 0
max_column  = 0

def crossover(first_parent, second_parent):
    child_pieces    = {}
    taken_positions = set()

    # Initialize with root piece
    root = first_parent.pieces[int(random.uniform(0, len(first_parent.pieces)))].id
    child_pieces[root] = (0, 0)
    taken_positions.add((0, 0))

    boundaries = available_boundaries(root, (0, 0), taken_positions, first_parent.rows, first_parent.columns)

    while len(child_pieces) < len(first_parent.pieces):
        (shared, position), bnd = get_shared_piece(boundaries, first_parent, second_parent, child_pieces, taken_positions)

        if shared != None:
            child_pieces[shared] = position
            taken_positions.add(position)
            boundaries.remove(bnd)
            boundaries.extend(available_boundaries(shared, position, taken_positions, first_parent.rows, first_parent.columns))
            continue

        (buddy, position), bnd = get_buddy_piece(boundaries, first_parent, second_parent, child_pieces, taken_positions)

        if buddy != None:
            child_pieces[buddy] = position
            taken_positions.add(position)
            boundaries.remove(bnd)
            boundaries.extend(available_boundaries(buddy, position, taken_positions, first_parent.rows, first_parent.columns))
            continue

        (best_match, position), bnd = get_best_match(boundaries, child_pieces, taken_positions)
        child_pieces[best_match] = position
        taken_positions.add(position)
        boundaries.remove(bnd)
        boundaries.extend(available_boundaries(best_match, position, taken_positions, first_parent.rows, first_parent.columns))

    pieces = [None] * (first_parent.rows * first_parent.columns)

    for piece, (row, column) in child_pieces.iteritems():
        index = (row - min_row) * first_parent.columns + (column - min_column)
        pieces[index] = first_parent.piece_by_id(piece)

    return Individual(pieces, first_parent.rows, first_parent.columns, shuffle=False)

def get_shared_piece(boundaries, first_parent, second_parent, taken_pieces, taken_positions):
    pieces = []
    choosen_boundaries = []
    for idx, (piece, orientation, position) in enumerate(boundaries):
        shared = shared_edge((piece, orientation), first_parent, second_parent)
        if shared != None and not shared in taken_pieces and position not in taken_positions:
            pieces.append((shared, position))
            choosen_boundaries.append((piece, orientation, position))

    if len(pieces) > 0:
        index = int(random.uniform(0, len(pieces)))

        return pieces[index], choosen_boundaries[index]
    else:
        return (None, None), -1

def get_buddy_piece(boundaries, first_parent, second_parent, taken_pieces, taken_positions):
    pieces = []
    choosen_boundaries = []
    for idx, (piece, orientation, position) in enumerate(boundaries):
        buddy = buddy_edge((piece, orientation), first_parent, second_parent)
        if buddy != None and not buddy in taken_pieces and position not in taken_positions:
            pieces.append((buddy, position))
            choosen_boundaries.append((piece, orientation, position))

    if len(pieces) > 0:
        index = int(random.uniform(0, len(pieces)))

        return pieces[index], choosen_boundaries[index]
    else:
        return (None, None), -1

def get_best_match(boundaries, taken_pieces, taken_positions):
    for piece, orientation, position in boundaries:
        if not position in taken_positions:
            return (best_available_match((piece, orientation), taken_pieces), position), (piece, orientation, position)

def in_range(row, column, rows, columns):
    global min_row
    global min_column
    global max_row
    global max_column

    if abs(min(min_row, row)) + abs(max(max_row, row)) >= rows:
        return False

    if abs(min(min_column, column)) + abs(max(max_column, column)) >= columns:
        return False

    min_row = min(min_row, row)
    max_row = max(max_row, row)
    min_column = min(min_column, column)
    max_column = max(max_column, column)

    return True

def available_boundaries(piece, piece_position, taken_positions, rows, columns):
    if len(taken_positions) == rows * columns:
        return []

    row, column = piece_position
    boundaries  = []

    if not (row - 1, column) in taken_positions and in_range(row - 1, column, rows, columns):
        boundaries.append((piece, "T", (row - 1, column)))

    if not (row, column + 1) in taken_positions and in_range(row, column + 1, rows, columns):
        boundaries.append((piece, "R", (row, column + 1)))

    if not (row + 1, column) in taken_positions and in_range(row + 1, column, rows, columns):
        boundaries.append((piece, "D", (row + 1, column)))

    if not (row, column - 1) in taken_positions and in_range(row, column - 1, rows, columns):
        boundaries.append((piece, "L", (row, column - 1)))

    return boundaries

def shared_edge((piece, orientation), first_parent, second_parent):
    fp_edge = first_parent.edge(piece, orientation)
    sp_edge = second_parent.edge(piece, orientation)

    if sp_edge == fp_edge:
        return fp_edge

def buddy_edge((piece, orientation), first_parent, second_parent):
    first_buddy  = DissimilarityMeasureCache.best_match(piece, orientation)
    second_buddy = DissimilarityMeasureCache.best_match(first_buddy, complementary_orientation(orientation))

    if second_buddy == piece:

        fp_edge = first_parent.edge(piece, orientation)
        if fp_edge == first_buddy:
            return fp_edge

        sp_edge = second_parent.edge(piece, orientation)
        if sp_edge == first_buddy:
            return sp_edge

def best_available_match((piece, orientation), child_pieces):
    best_match = DissimilarityMeasureCache.best_match(piece, orientation)

    if best_match in child_pieces:
        popped_items = []

        # Get most compatible piece that is not already taken
        while best_match in child_pieces:
            best_match, value = DissimilarityMeasureCache.best_match_table[piece][orientation].popitem()
            popped_items.append((best_match, value))

        # Restore priority_queue back
        for key, value in popped_items:
            DissimilarityMeasureCache.best_match_table[piece][orientation].additem(key, value)

    return best_match

# Dictionary is defined outside functions for speed.
# This way its created once, not each tima on function call
_complementary_orientation = {
        "T": "D",
        "R": "L",
        "D": "T",
        "L": "R"
}

def complementary_orientation(orientation):
    return _complementary_orientation.get(orientation, None)

