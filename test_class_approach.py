from genetic_algorithm_classes_approach import *


brain_instance = BrainInitiation(
    nr_of_genes=15, nr_of_input=5, nr_of_actions=3, nr_of_inner=4
)
initial_genomes = brain_instance.generate_initial_genomes_for_population()
print(initial_genomes)

weight_calc = CalculateWeights(initial_genomes)
result = weight_calc.calculate_individual_output_weights()
print(result)
