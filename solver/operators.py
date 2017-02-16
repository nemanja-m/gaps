import random
import bisect
import heapq

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
    probability_intervals = [sum(fitnesses[:i+1]) for i in range(len(fitnesses))]

    random_select  = random.uniform(0, probability_intervals[-1])
    selected_index = bisect.bisect_left(probability_intervals, random_select)

    return population[selected_index]

def crossover(first_parent, second_parent):
    child_pieces    = {}
    available_edges = []
    visited_nodes   = {}

    fp_edges = {}
    sp_edges = {}

    current_piece = first_parent.pieces[0]

    rows, columns = 1, 1
    child_pieces[current_piece.id] = (0, 0)

    # Parents edges are maps where keys are pieces IDs and values are maps with edge's IDs
    # as a key and edge position as a value (Top, Right, Down, Left)
    fp_edges[current_piece.id] = possible_edges(current_piece.id, first_parent, visited_edges)
    sp_edges[current_piece.id] = possible_edges(current_piece.id, second_parent, visited_edges)

    # We are looking for an edge appearing in both parents
    edge, piece = select_shared_edge(fp_edges, sp_edges)

    edge, piece = select_best_buddy_edge(fp_edges sp_edges)

    # TODO Remove selected edge from pool ov edges
    # TODO Remove piece from dictionary if it doesn't have any more edges left

    if (edge, piece) != (None, None):
        relative_position = child_pieces[piece]

        offset = 0

        if position == "T":
            offset = (-1, 0)
        elif position == "R":
            offset = (0, 1)
        elif position == "D":
            offset = (1, 0)
        else:
            offset = (0, -1)

        # New position for selected edge
        child_pieces[edge] = tuple(map(sum, zip(relative_position, offset)))

        fp_edges[edge] = possible_edges(edge, first_parent, child_pieces)
        sp_edges[edge] = possible_edges(edge, second_parent, child_pieces)

    return edges

def select_shared_edge(fp_edges, sp_edges):
    """Finds shared edges between parents"""
    for piece, edges in fp_edges.iteritems():
        for edge, position in edges.iteritems():
            # Edge is present in both parents
            if edge in sp_edges[piece] and sp_edges[piece][edge] == position:
                return edge, piece

    return None, None

def select_best_buddy_edge(first_parent, fp_edges, second_parent, sp_edges):
    """Finds best-buddy edge in curent kernel"""

    # Looking in first parent
    for piece, edges in fp_edges.iteritems():
        for edge, position in edges.iteritems():
            # Given pieces are best buddies
            if first_parent.best_buddies(piece, edge):
                return edge, piece

    # Looking in second parent
    for piece, edges in sp_edges.iteritems():
        for edge, position in edges.iteritems():
            # Given pieces are best buddies
            if second_parent.best_buddies(piece, edge):
                return edge, piece

    return None, None

def possible_edges(piece, parent, visited_edges):
    """Returns dictionary where keys are piece IDs and values are edge orientation

    Piece's IDs that are not previously visited are returned as a dictionary.

    i.e.

    {
        1: "L",
        2: "R"
    }

    :param piece: Source piece for selected edges.
    :param parent: Individual containing given pieces.
    :param visited_edges: Dictionary with information about visited edges.

    Usage::

        >>> possible_edges(1, parent, visited_edges)
        >>> { 2: "R", 42: "T" }

    """

    edges = [edge for edge in parent.edges(piece) if not edge[1] in visited_edges]
    return { edge[1]: edge[2] for edge in edges }

