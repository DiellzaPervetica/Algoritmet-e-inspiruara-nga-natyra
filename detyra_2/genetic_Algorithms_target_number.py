import os
import random
import matplotlib.pyplot as plt

DIGITS = list(range(10))
OPERATORS = ['+', '-', '*']

def individual_to_string(individual):
    return ' '.join(str(gene) for gene in individual)

def evaluate(individual):
    result = individual[0]

    for i in range(1, len(individual), 2):
        op = individual[i]
        num = individual[i + 1]

        if op == '+':
            result = result + num
        elif op == '-':
            result = result - num
        elif op == '*':
            result = result * num

    return result

def initialize_population(pop_size, num_genes):
    population = []

    for _ in range(pop_size):
        individual = []
        for i in range(num_genes):
            if i % 2 == 0:
                individual.append(random.choice(DIGITS))
            else:
                individual.append(random.choice(OPERATORS))
        population.append(individual)

    return population

def fitness(individual, target):
    value = evaluate(individual)
    return 1 / (1 + abs(target - value))

def select(population, fitnesses):
    selected = random.sample(list(zip(population, fitnesses)), 3)
    selected.sort(key=lambda x: x[1], reverse=True)
    return selected[0][0]

def crossover(parent1, parent2):
    point = random.randint(1, len(parent1) - 2)
    child = parent1[:point] + parent2[point:]
    return child

def mutate(individual, mutation_rate):
    new_individual = individual[:]

    for i in range(len(new_individual)):
        if random.random() < mutation_rate:
            if i % 2 == 0:
                new_individual[i] = random.choice(DIGITS)
            else:
                new_individual[i] = random.choice(OPERATORS)

    return new_individual

def run_ga(target, pop_size, num_genes, generations, mutation_rate):
    population = initialize_population(pop_size, num_genes)

    best_fitness_per_generation = []
    best_individual_ever = None
    best_value_ever = None
    best_score_ever = -1

    for gen in range(generations):
        fitnesses = [fitness(ind, target) for ind in population]

        best_fitness = max(fitnesses)
        best_index = fitnesses.index(best_fitness)
        best_individual = population[best_index]
        best_value = evaluate(best_individual)

        best_fitness_per_generation.append(best_fitness)

        if best_fitness > best_score_ever:
            best_score_ever = best_fitness
            best_individual_ever = best_individual[:]
            best_value_ever = best_value

        print(f"Gen {gen}: {individual_to_string(best_individual)} = {best_value} | Fitness: {best_fitness:.6f}")

        if best_value == target:
            print("Perfect solution found!")
            break

        new_population = []

        for _ in range(pop_size):
            parent1 = select(population, fitnesses)
            parent2 = select(population, fitnesses)
            child = crossover(parent1, parent2)
            child = mutate(child, mutation_rate)
            new_population.append(child)

        population = new_population

    return best_individual_ever, best_value_ever, best_fitness_per_generation

