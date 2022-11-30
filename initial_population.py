def initial_population(nr_individuals, nr_of_genes, nr_of_input, nr_of_actions, nr_of_inner, world_size):
    '''generates list of individuals with genome and brain'''
    individuals = generate_initial_genomes_for_population(nr_individuals, nr_of_genes, nr_of_input, nr_of_actions, nr_of_inner)

    ## initial brain and position generator
    result = calculate_individual_output_weights(individuals)

    ## add genome
    for indiv in result:
        result[indiv]['genome'] = [i.hex_id for i in individuals[indiv]]
        
    ## assign position remove brains without output
    pos = generate_random_coords(world_size, nr_individuals)
    assign_position_and_remove_outputless_brains(result, pos)
    return result