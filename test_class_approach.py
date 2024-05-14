from genetic_algorithm_classes_approach import *

nr_individuals = 10
world_size = 30

population = {}

for nr in range(nr_individuals):

    brain_instance = BrainInitiation(
        nr_of_genes=15, nr_of_input=5, nr_of_actions=3, nr_of_inner=4
    )
    initial_genomes = brain_instance.generate_initial_genomes_for_population()
    # print(initial_genomes)
    # print()
    weight_calc = CalculateWeights(initial_genomes)
    result_1 = weight_calc.calculate_individual_output_weights()
    # print(result)
    population[nr] = result_1

pop_param = Population(world_size=world_size, nr_individuals=10, population=population)
result_2 = pop_param.clear_malfunction_brain_and_assign_position()

pop_move = PopulationMovement(world_size=world_size, population=result_2)
result_3 = pop_move.steps_in_generation()

print(result_3)
