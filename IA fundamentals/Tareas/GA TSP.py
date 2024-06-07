"""
    solution =  cromosomas, genotipo o solucion

    3 tipo de initialization:
    1.- Random initialization, diversity OK
    2.- Heuristic initialization, intensify can speed up convergence
    3.- Hybrid initialization, half random and half heuristic
"""

import numpy as np
from scipy.spatial.distance import pdist, squareform
import random
import Plotting

N = 5
pop_size = 8
cities = np.random.rand(N, 2)
distances = squareform(pdist(cities, 'euclidean'))

""" RANDOM INITIALIZATION """
def create_initial_population(pop_size: int, num_cities: int) -> list:
    population = []
    for _ in range(pop_size):
        individual = list(np.random.permutation(num_cities))
        population.append(individual)
        #print('Initial population:', individual)
    print(type(population))

    return population

def fitness(solution: list) -> float:
    distance = 0
    for i in range(len(solution)-1):
        distance += distances[solution[i], solution[i+1]]
    return distance

population = create_initial_population(pop_size, N)
print(population)
Plotting.plot_tsp(cities, population[0])
print(fitness(population[0]))

def roulette_wheel_selection(population: list, all_fitness: list) -> list:
    total_fitness = sum(all_fitness)
    selection_probs = [f/total_fitness for f in all_fitness]

    return population[np.random.choice(len(population), p=selection_probs)]

print("Population Size: ", len(population))
print("Num. cities: ", N)
print(population)

all_fitness = [fitness(individual) for individual in population]
print(all_fitness, "\n")








