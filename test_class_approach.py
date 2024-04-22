from genetic_algorithm_classes_approach import Brain


brain_instance = Brain(nr_of_genes=10, nr_of_input=5, nr_of_actions=3, nr_of_inner=4)
initial_genomes = brain_instance.generate_initial_genomes_for_population()
print(initial_genomes)
