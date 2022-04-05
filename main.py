import numpy as np, random, operator, pandas as pd, matplotlib.pyplot as plt
from Fitness import Fitness
import cv2
import time

HEIGHT = 600
WIDTH = 800
FONT = cv2.FONT_HERSHEY_COMPLEX
SIZE = 0.7
WHITE = (255, 255, 255)
RED = (0, 0, 255)


# Generate random in (x, y) positions
def create_cities(width, height, count):
    cities = []
    for i in range(count):
        position_x = np.random.randint(width)
        position_y = np.random.randint(height)
        cities.append((position_x, position_y))
    return cities

    # Fixed to compare
    # return [(779, 74), (403, 343), (551, 395), (525, 519), (346, 526), (604, 264)]
    # return [(91, 53), (201, 370), (557, 87), (238, 145), (458, 158), (438, 508), (302, 23), (263, 385), (585, 298), (53, 116), (61, 507), (171, 491)]


# Create a random route
def create_random_route(cities):
    route = random.sample(cities, len(cities))
    return route


# Generate initial population
def initial_population(num_cities, cities):
    population = []

    for i in range(num_cities):
        population.append(create_random_route(cities))

    return population


# Calculate the fitness of every individual and return sorted by the best to the worst
def rank_routes(population):
    fitnessResults = {}
    for i in range(len(population)):
        fitnessResults[i] = Fitness(population[i]).route_fitness()

    return sorted(fitnessResults.items(), key=operator.itemgetter(1), reverse=True)


# Save the best 'elite_size' individuals and sort the rest based on the capability(some are repeated)
def selection(pop_ranked, elite_size):
    selection_results = []

    # Calculate the relative fitness by position
    df = pd.DataFrame(np.array(pop_ranked), columns=['Index', 'Fitness'])
    df['cum_sum'] = df.Fitness.cumsum()
    df['cum_perc'] = 100 * df.cum_sum / df.Fitness.sum()

    for i in range(elite_size):
        selection_results.append(pop_ranked[i][0])

    for i in range(len(pop_ranked) - elite_size):
        pick = 100 * random.random()

        # Pick if the probability is lower than the pick number
        for i in range(len(pop_ranked)):
            if pick <= df.iat[i, 3]:
                selection_results.append(pop_ranked[i][0])
                break

    return selection_results


# Get the population data
def mating_pool(population, selection_results):
    mating_pool_list = []
    for index in selection_results:
        mating_pool_list.append(population[index])

    return mating_pool_list


# Get a random selected subsets from the first parent, and then fill the rest with genes from the parent 2 in order
def crossover(parent_1, parent_2):
    child_p1 = []

    gene_a = int(random.random() * len(parent_1))
    gene_b = int(random.random() * len(parent_1))

    start_gene = min(gene_a, gene_b)
    end_gene = max(gene_a, gene_b)

    for i in range(start_gene, end_gene):
        child_p1.append(parent_1[i])

    child_p2 = [item for item in parent_2 if item not in child_p1]

    return child_p1 + child_p2


# Sort, then select one individual from the start and another from the end to do the crossover
def breed_population(mating_pool, elite_size):
    children = []
    lenght = len(mating_pool) - elite_size
    pool = random.sample(mating_pool, len(mating_pool))

    for i in range(elite_size):
        children.append(mating_pool[i])

    for i in range(lenght):
        child = crossover(pool[i], pool[len(mating_pool) - i - 1])
        children.append(child)

    return children


# Here we provide a low mutation_rate to randomly swap 2 individual to avoid local convergence
def mutate(individual, mutation_rate):
    for swapped in range(len(individual)):
        if random.random() < mutation_rate:
            swap_with = int(random.random() * len(individual))

            city_1 = individual[swapped]
            city_2 = individual[swap_with]

            individual[swapped] = city_2
            individual[swap_with] = city_1

    return individual


# call mutate function for every individual
def mutate_population(population, mutation_rate):
    mutated_pop = []

    for i in range(len(population)):
        mutated_ind = mutate(population[i], mutation_rate)
        mutated_pop.append(mutated_ind)

    return mutated_pop


def next_generation(current_gen, elite_size, mutation_rate):
    pop_ranked = rank_routes(current_gen)
    selection_results = selection(pop_ranked, elite_size)
    mating_pool_list = mating_pool(current_gen, selection_results)
    children = breed_population(mating_pool_list, elite_size)
    next_generation = mutate_population(children, mutation_rate)

    # draw(WIDTH, HEIGHT, current_gen[0])

    return next_generation


def genetic_algorithm(population, pop_size, elite_size, mutation_rate, generations):
    pop = initial_population(pop_size, population)
    print("Initial distance: " + str(1 / rank_routes(pop)[0][1]))       #Get the first individual distance

    for i in range(0, generations):
        pop = next_generation(pop, elite_size, mutation_rate)

    print("Final distance: " + str(1 / rank_routes(pop)[0][1]))
    best_route_index = rank_routes(pop)[0][0]
    best_route = pop[best_route_index]

    print("Best Score: {}".format(str(1 / rank_routes(pop)[0][1])))
    print("Best solution: {}".format(best_route))

    return best_route


def genetic_algorithm_plot(population, pop_size, elite_size, mutation_rate, generations):
    pop = initial_population(pop_size, population)
    progress = []
    progress.append(1 / rank_routes(pop)[0][1])

    for i in range(generations):
        pop = next_generation(pop, elite_size, mutation_rate)
        progress.append(1 / rank_routes(pop)[0][1])

    plt.plot()
    plt.ylabel('Distance')
    plt.xlabel('Generation')
    plt.show()


def draw(width, height, cities):
    frame = np.zeros((height, width, 3))
    for i in range(len(cities)):
        point_a = (cities[i][0], cities[i][1])
        point_b = (cities[i - 1][0], cities[i - 1][1])
        cv2.line(frame, point_a, point_b, WHITE, 2)
    for city in cities:
        cv2.circle(frame, (city[0], city[1]), 5, RED, -1)
    cv2.putText(frame, f"Score", (25, 50), FONT, SIZE, WHITE)
    cv2.putText(frame, f"Best Score", (25, 75), FONT, SIZE, WHITE)
    cv2.putText(frame, f"Worst Score", (25, 100), FONT, SIZE, WHITE)
    # cv2.putText(frame, f": {infos[0]:.2f}", (175, 50), FONT, SIZE, WHITE)
    # cv2.putText(frame, f": {infos[1]:.2f}", (175, 75), FONT, SIZE, WHITE)
    # cv2.putText(frame, f": {infos[2]:.2f}", (175, 100), FONT, SIZE, WHITE)
    cv2.imshow("Genetic Algorithm Traveling Salesman", frame)
    cv2.waitKey(5)


if __name__ == '__main__':
    num_cities = int(input("Number of cities: "))

    cities = create_cities(WIDTH, HEIGHT, num_cities)

    t1 = time.time()

    genetic_algorithm(population=cities, pop_size=100, elite_size=30, mutation_rate=0.01, generations=200)

    t2 = time.time()

    total_time = t2 - t1

    print("Total time: {}s".format(total_time))