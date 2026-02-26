import os
import random
import string
import matplotlib.pyplot as plt

def initialize_population(pop_size, length, charset):
    population = []
    for _ in range(pop_size):
        individual = ''.join(random.choice(charset) for _ in range(length))
        population.append(individual)
    return population

def fitness(individual, target):
    score = 0
    for i in range(len(target)):
        if individual[i] == target[i]:
            score += 1
    return score

def select(population, fitnesses):
    selected = random.sample(list(zip(population, fitnesses)), 3)
    selected.sort(key=lambda x: x[1], reverse=True)
    return selected[0][0]

def crossover(parent1, parent2):
    point = random.randint(1, len(parent1) - 1)
    child = parent1[:point] + parent2[point:]
    return child

def mutate(individual, mutation_rate, charset):
    new_individual = ""
    for char in individual:
        if random.random() < mutation_rate:
            new_individual += random.choice(charset)
        else:
            new_individual += char
    return new_individual

def run_ga(target, pop_size, generations, mutation_rate, charset):
    length = len(target)
    population = initialize_population(pop_size, length, charset)

    best_fitness_per_generation = []

    for gen in range(generations):
        fitnesses = [fitness(ind, target) for ind in population]

        best_fitness = max(fitnesses)
        best_individual = population[fitnesses.index(best_fitness)]

        print(f"Gen {gen}: {best_individual} | Fitness: {best_fitness}")

        best_fitness_per_generation.append(best_fitness)

        if best_fitness == length:
            print("Perfect solution found!")
            break

        new_population = []

        for _ in range(pop_size):
            parent1 = select(population, fitnesses)
            parent2 = select(population, fitnesses)
            child = crossover(parent1, parent2)
            child = mutate(child, mutation_rate, charset)
            new_population.append(child)

        population = new_population

    return best_fitness_per_generation

def run_experiments(target="EVOLUTION", charset=string.ascii_uppercase):
    generations = 200

    experiments = [
        ("baseline_pop100_mut0.01", 100, 0.01, 1),
        ("mut0",                    100, 0.0,  1),
        ("mut0.5",                  100, 0.5,  1),
        ("pop5_mut0.01",            5,   0.01, 1),
        ("baseline_run3x",          100, 0.01, 3),
    ]

    # krijo folderin për eksperimente
    exp_folder = "results/experiments"
    os.makedirs(exp_folder, exist_ok=True)

    for name, pop_size, mutation_rate, runs in experiments:
        for r in range(runs):
            print("\n" + "=" * 40)
            print(f"Experiment: {name} | run {r+1}")

            fitness_curve = run_ga(target, pop_size, generations, mutation_rate, charset)

            plt.figure()
            plt.plot(fitness_curve)
            plt.xlabel("Generation")
            plt.ylabel("Best Fitness")
            plt.ylim(0, len(target))
            plt.title(f"{target} | {name} | run {r+1}")

            safe_target = target.replace(" ", "_")
            filename = f"{safe_target}__{name}__run{r+1}.png"
            filepath = os.path.join(exp_folder, filename)

            plt.savefig(filepath)
            plt.close()

def main():

    solutions_folder = "results/solutions"
    experiments_folder = "results/experiments"
    os.makedirs(solutions_folder, exist_ok=True)
    os.makedirs(experiments_folder, exist_ok=True)

    test_cases = [
        ("HI", string.ascii_uppercase),
        ("EVOLUTION", string.ascii_uppercase),
        ("NATURE INSPIRED", string.ascii_uppercase + " "),
        ("AAAAAAAAAA", string.ascii_uppercase),
        ("GA IS COOL", string.ascii_uppercase + " "),
    ]

    for target, charset in test_cases:
        print("\n" + "=" * 40)
        print(f"Running GA for target: {target}")

        fitness_curve = run_ga(target, 100, 200, 0.01, charset)

        plt.figure()
        plt.plot(fitness_curve)
        plt.xlabel("Generation")
        plt.ylabel("Best Fitness")
        plt.ylim(0, len(target))
        plt.title(target)

        filename = target.replace(" ", "_") + ".png"
        filepath = os.path.join(solutions_folder, filename)

        plt.savefig(filepath)
        plt.close()

    run_experiments("EVOLUTION", string.ascii_uppercase)
    run_experiments("NATURE INSPIRED", string.ascii_uppercase + " ")

if __name__ == "__main__":
    main()