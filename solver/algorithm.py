from solver import helpers
from solver import fitness
from solver.operators  import select
from solver.operators  import crossover
from solver.individual import Individual

def create_population(problem, population_size):
    return [Individual(problem) for i in range(population_size)]

def start_evolution(problem, population_size=1000, generations=100):
    population = create_population(population_size)

    for generation in range(generations):
        new_population = []

        # Parallelize this !
        map(fitness.evaluate, population)

        while len(new_population) <= population_size:
            first_parent  = select(population)
            second_parent = select(population)
            child         = crossover(first_parent, second_parent)

            new_population.append(child)

        population = new_population

    return best_individual(population)

