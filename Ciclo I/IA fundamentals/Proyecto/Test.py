import numpy as np
import random
from typing import List, Tuple
from dataclasses import dataclass
import time
import pandas as pd

@dataclass
class Customer:
    id: int
    x: float
    y: float
    demand: float
    ready_time: float
    due_time: float
    service_time: float

@dataclass
class Vehicle:
    capacity: float

class VRPTW_GA:
    def __init__(self, customers: List[Customer], max_vehicles: int, vehicle_capacity: float, depot: Customer,
                 population_size: int = 300, generations: int = 5000,
                 mutation_rate: float = 0.02, crossover_rate: float = 0.8,
                 elitism_rate: float = 0.1):
        self.customers = customers
        self.max_vehicles = max_vehicles
        self.vehicle_capacity = vehicle_capacity
        self.depot = depot
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism_rate = elitism_rate
        self.distance_matrix = self.create_distance_matrix()
        self.fitness_history = []

    def create_distance_matrix(self):
        n = len(self.customers) + 1  # +1 for depot
        matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i+1, n):
                if i == 0:
                    dist = self.euclidean_distance(self.depot, self.customers[j-1])
                elif j == 0:
                    dist = self.euclidean_distance(self.customers[i-1], self.depot)
                else:
                    dist = self.euclidean_distance(self.customers[i-1], self.customers[j-1])
                matrix[i][j] = matrix[j][i] = dist
        return matrix

    def initialize_population(self):
        population = []
        for _ in range(self.population_size):
            chromosome = self.generate_initial_solution()
            population.append(chromosome)
        return population

    def generate_initial_solution(self):
        unassigned = list(range(1, len(self.customers) + 1))
        routes = []
        for _ in range(self.max_vehicles):
            if not unassigned:
                break
            route = []
            capacity_left = self.vehicle_capacity
            time = 0
            current = 0  # Start from depot

            while unassigned:
                feasible = [c for c in unassigned if self.customers[c-1].demand <= capacity_left]
                if not feasible:
                    break

                next_customer = min(feasible, key=lambda c: (
                    max(time + self.distance_matrix[current][c], self.customers[c-1].ready_time)
                    + self.distance_matrix[c][0]
                ))

                arrival = max(time + self.distance_matrix[current][next_customer], self.customers[next_customer-1].ready_time)
                if arrival > self.customers[next_customer-1].due_time:
                    break

                route.append(next_customer)
                unassigned.remove(next_customer)
                capacity_left -= self.customers[next_customer-1].demand
                time = arrival + self.customers[next_customer-1].service_time
                current = next_customer

            if route:
                routes.append(route)

        return [customer for route in routes for customer in route]

    def fitness(self, chromosome):
        routes = self.decode_chromosome(chromosome)
        total_distance = 0
        total_violations = 0
        vehicles_used = len(routes)

        for route in routes:
            route_distance = 0
            current_time = 0
            current_load = 0
            prev_customer = 0  # Depot

            for customer_id in route:
                customer = self.customers[customer_id-1]
                distance = self.distance_matrix[prev_customer][customer_id]
                route_distance += distance
                current_time = max(current_time + distance, customer.ready_time)

                if current_time > customer.due_time:
                    total_violations += current_time - customer.due_time

                current_time += customer.service_time
                current_load += customer.demand

                if current_load > self.vehicle_capacity:
                    total_violations += current_load - self.vehicle_capacity

                prev_customer = customer_id

            route_distance += self.distance_matrix[prev_customer][0]  # Return to depot
            total_distance += route_distance

        vehicle_penalty = max(0, vehicles_used - self.max_vehicles) * 1000

        return 1 / (1 + total_distance + total_violations * 100 + vehicle_penalty)

    def decode_chromosome(self, chromosome):
        routes = []
        current_route = []
        current_load = 0
        current_time = 0
        prev_customer = 0  # Depot

        for customer_id in chromosome:
            customer = self.customers[customer_id-1]
            distance = self.distance_matrix[prev_customer][customer_id]
            arrival_time = max(current_time + distance, customer.ready_time)

            if (current_load + customer.demand <= self.vehicle_capacity and
                arrival_time <= customer.due_time):
                current_route.append(customer_id)
                current_load += customer.demand
                current_time = arrival_time + customer.service_time
                prev_customer = customer_id
            else:
                if current_route:
                    routes.append(current_route)
                current_route = [customer_id]
                current_load = customer.demand
                current_time = self.distance_matrix[0][customer_id] + customer.service_time
                prev_customer = customer_id

        if current_route:
            routes.append(current_route)

        return routes

    def euclidean_distance(self, customer1, customer2):
        return np.sqrt((customer1.x - customer2.x)**2 + (customer1.y - customer2.y)**2) / 40

    def selection(self, population, fitnesses):
        tournament_size = 3
        selected = []
        for _ in range(2):
            competitors = random.sample(list(enumerate(fitnesses)), tournament_size)
            winner = max(competitors, key=lambda x: x[1])[0]
            selected.append(population[winner])
        return selected

    def crossover(self, parent1, parent2):
        if random.random() < self.crossover_rate:
            crossover_point1 = random.randint(0, len(parent1) - 1)
            crossover_point2 = random.randint(crossover_point1, len(parent1))

            child1 = parent1[:crossover_point1] + [gene for gene in parent2 if gene not in parent1[:crossover_point1]]
            child2 = parent2[:crossover_point1] + [gene for gene in parent1 if gene not in parent2[:crossover_point1]]

            return child1, child2
        return parent1, parent2

    def mutate(self, chromosome):
        for i in range(len(chromosome)):
            if random.random() < self.mutation_rate:
                j = random.randint(0, len(chromosome) - 1)
                chromosome[i], chromosome[j] = chromosome[j], chromosome[i]
        return chromosome

    def local_search(self, chromosome):
        routes = self.decode_chromosome(chromosome)
        improved = True
        while improved:
            improved = False
            for i, route in enumerate(routes):
                for j in range(len(route)):
                    for k in range(j+1, len(route)):
                        new_route = route[:j] + route[j:k+1][::-1] + route[k+1:]
                        if self.is_route_feasible(new_route) and self.route_cost(new_route) < self.route_cost(route):
                            routes[i] = new_route
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
        return [customer for route in routes for customer in route]

    def is_route_feasible(self, route):
        current_load = 0
        current_time = 0
        prev_customer = 0  # Depot

        for customer_id in route:
            customer = self.customers[customer_id-1]
            distance = self.distance_matrix[prev_customer][customer_id]
            arrival_time = max(current_time + distance, customer.ready_time)

            if arrival_time > customer.due_time or current_load + customer.demand > self.vehicle_capacity:
                return False

            current_load += customer.demand
            current_time = arrival_time + customer.service_time
            prev_customer = customer_id

        return True

    def route_cost(self, route):
        cost = 0
        prev_customer = 0  # Depot
        for customer_id in route:
            cost += self.distance_matrix[prev_customer][customer_id]
            prev_customer = customer_id
        cost += self.distance_matrix[prev_customer][0]  # Return to depot
        return cost

    def run(self):
        population = self.initialize_population()
        best_fitness = float('-inf')
        best_solution = None

        self.fitness_history = []

        for generation in range(self.generations):
            fitnesses = [self.fitness(chromosome) for chromosome in population]
            best_index = np.argmax(fitnesses)

            if fitnesses[best_index] > best_fitness:
                best_fitness = fitnesses[best_index]
                best_solution = population[best_index]
                print(f"Generation {generation}: Best Fitness = {best_fitness}")

            self.fitness_history.append(best_fitness)

            elite_size = int(self.elitism_rate * self.population_size)
            elite = sorted(zip(population, fitnesses), key=lambda x: x[1], reverse=True)[:elite_size]

            new_population = [chromosome for chromosome, _ in elite]

            while len(new_population) < self.population_size:
                parent1, parent2 = self.selection(population, fitnesses)
                child1, child2 = self.crossover(parent1, parent2)
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                child1 = self.local_search(child1)
                child2 = self.local_search(child2)
                new_population.extend([child1, child2])

            population = new_population[:self.population_size]

        return best_solution, best_fitness

    def get_solution_metrics(self, solution):
        routes = self.decode_chromosome(solution)
        total_distance = 0
        total_vehicles = len(routes)
        total_customers_served = sum(len(route) for route in routes)

        for route in routes:
            route_distance = 0
            prev_customer = 0  # Depot

            for customer_id in route:
                route_distance += self.distance_matrix[prev_customer][customer_id]
                prev_customer = customer_id

            route_distance += self.distance_matrix[prev_customer][0]  # Return to depot
            total_distance += route_distance

        return {
            "total_distance": total_distance,
            "total_vehicles": total_vehicles,
            "total_customers_served": total_customers_served,
            "average_distance_per_vehicle": total_distance / total_vehicles if total_vehicles > 0 else 0
        }

def load_solomon_dataset_excel(filename: str) -> Tuple[List[Customer], Customer]:
    data = pd.read_excel(filename)
    data.dropna(axis=1, how='all', inplace=True)
    data.drop(columns=data.columns[data.count() <= 150], inplace=True)

    data = data[(data['Estado'] == 'No entregado')]
    data['Indice'] = list(range(data.shape[0]))
    data['Fecha inicio'] = pd.to_datetime(data['Fecha inicio']).apply(lambda x: x.timestamp()) / 3600
    data['Fecha fin'] = pd.to_datetime(data['Fecha fin']).apply(lambda x: x.timestamp()) / 3600
    data['Tiempo Servicio'] = data['Tiempo Servicio'] / 60
    data = data[['Indice', 'Latitud dirección', 'Longitud dirección', 'Demanda', 'Fecha inicio', 'Fecha fin',
                 'Tiempo Servicio']]
    customers = []
    depot = None

    for _, row in data.iterrows():
        customer_id = int(row[0])
        x = float(row[1])
        y = float(row[2])
        demand = float(row[3])
        ready_time = float(row[4])
        due_time = float(row[5])
        service_time = float(row[6])

        customer = Customer(customer_id, x, y, demand, ready_time, due_time, service_time)

        if customer_id == 1:
            depot = customer
        else:
            customers.append(customer)

    if depot is None:
        raise ValueError("Depot not found in the dataset")

    return customers, depot

def main():
    instance_name = "Nirex"  # Change this according to the instance you're using
    filename = f"{instance_name}.xlsx"  # Make sure the file is in the correct directory

    # Configuration
    vehicle_capacities = [5, 10, 15, 20]  # List of vehicle capacities to test
    vehicle_counts_to_test = [10]  # Maximum number of vehicles for the test

    try:
        customers, depot = load_solomon_dataset_excel(filename)
    except Exception as e:
        print(f"Error loading Solomon dataset: {e}")
        return

    results = []
    fitness_history_data = []

    max_vehicles = max(vehicle_counts_to_test)  # Use the maximum value from vehicle_counts_to_test

    for vehicle_capacity in vehicle_capacities:
        print(f"\nRunning with maximum {max_vehicles} vehicles and capacity {vehicle_capacity}:")

        ga = VRPTW_GA(customers, max_vehicles, vehicle_capacity, depot,
                      population_size=600,
                      generations=2000,
                      mutation_rate=0.005,
                      crossover_rate=0.2,
                      elitism_rate=0.001)

        start_time = time.time()
        best_solution, best_fitness = ga.run()
        end_time = time.time()

        metrics = ga.get_solution_metrics(best_solution)
        print("\nSolution found:")
        print(f"Fitness: {best_fitness}")
        print(f"Total distance: {metrics['total_distance']:.2f}")
        print(f"Vehicles used: {metrics['total_vehicles']}")
        print(f"Customers served: {metrics['total_customers_served']}")
        print(f"Average distance per vehicle: {metrics['average_distance_per_vehicle']:.2f}")
        print(f"Execution time: {end_time - start_time:.2f} seconds")

        results.append({
            'max_vehicles': max_vehicles,
            'vehicle_capacity': vehicle_capacity,
            'best_fitness': best_fitness,
            'total_distance': metrics['total_distance'],
            'vehicles_used': metrics['total_vehicles'],
            'customers_served': metrics['total_customers_served'],
            'avg_distance_per_vehicle': metrics['average_distance_per_vehicle'],
            'execution_time': end_time - start_time
        })

        # Add fitness history data
        for generation, fitness in enumerate(ga.fitness_history):
            fitness_history_data.append({
                'max_vehicles': max_vehicles,
                'vehicle_capacity': vehicle_capacity,
                'generation': generation,
                'fitness': fitness
            })

    # Export results to CSV
    df_results = pd.DataFrame(results)
    df_results.to_csv(f'{instance_name}_results_varying_capacity.csv', index=False)

    # Export fitness history to CSV
    df_fitness_history = pd.DataFrame(fitness_history_data)
    df_fitness_history.to_csv(f'{instance_name}_fitness_history.csv', index=False)

if __name__ == "__main__":
    main()