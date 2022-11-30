def hexval_to_bin(gene):
    binary = bin(int(gene, 16))[2:]
    if len(binary) <= 32:
        factor = 32 - len(binary)
        binary = '0' * factor + binary
        return binary

def split_genome(hexval):
       
    binary = hexval_to_bin(hexval)    
    source_type, source_id = binary[0], binary[1:8]
    sink_type, sink_id = binary[8], binary[9:16]
    weight_sign, weight = binary[16], binary[17:]

    gene_value_list = {key:int(val, 2) for key, val in zip(['source_type', 'source_id', 'sink_type', 'sink_id', 'weight_sign', 'weight'], [source_type, source_id, sink_type, sink_id, weight_sign, weight])}

    return gene_value_list

def get_neurons_body(gen_component, nr_of_input, nr_of_inner, nr_of_actions):
    if gen_component['source_type'] == 0:
        input_id = gen_component['source_id'] % nr_of_input
        input_type = 'in'
    elif gen_component['source_type'] == 1:
        input_id = gen_component['source_id'] % nr_of_inner
        input_type = 'mid'

    if gen_component['sink_type'] == 0:
        output_id = gen_component['sink_id'] % nr_of_inner
        output_type = 'mid'
    elif gen_component['sink_type'] == 1:
        output_id = gen_component['sink_id'] % nr_of_actions
        output_type = 'out'

    if gen_component['weight_sign'] == 0:
        weight = (gen_component['weight'] /8191.25) * -1
    elif gen_component['weight_sign'] == 1:
        weight = gen_component['weight'] / 8191.25

    differ_neuron = f'{input_type}{input_id}{output_type}{output_id}'
    
    return input_id, input_type, weight, output_id, output_type, differ_neuron

def gene_to_neuron(hexa_list, nr_of_input, nr_of_actions, nr_of_inner):
    gene_translated = []
    for l, hex_id in enumerate(hexa_list):
        gen_component = split_genome(hex_id)
        
        input_id, input_type, weight, output_id, output_type, differ_neuron = get_neurons_body(gen_component, nr_of_input, nr_of_inner, nr_of_actions)
        
        l = Neuron(hex_id, input_id, input_type, weight, output_id, output_type, differ_neuron)
        gene_translated.append(l)
    return gene_translated
    
def generate_initial_genomes_for_population(nr_individuals, nr_of_genes, nr_of_input, nr_of_actions, nr_of_inner):
    dic = {}
    for nr_idividual in range(nr_individuals):
        
        hexa_list = [token_hex(4) for i in range(nr_of_genes)]
        dic[nr_idividual] = gene_to_neuron(hexa_list, nr_of_input, nr_of_actions, nr_of_inner)

    return dic       


def gene_to_neuron(hexa_list, nr_of_input, nr_of_actions, nr_of_inner):
    gene_translated = []
    for l, hex_id in enumerate(hexa_list):
        gen_component = split_genome(hex_id)
        
        input_id, input_type, weight, output_id, output_type, differ_neuron = get_neurons_body(gen_component, nr_of_input, nr_of_inner, nr_of_actions)
        
        l = Neuron(hex_id, input_id, input_type, weight, output_id, output_type, differ_neuron)
        gene_translated.append(l)
    return gene_translated

def generate_initial_genomes_for_population(nr_individuals, nr_of_genes, nr_of_input, nr_of_actions, nr_of_inner):
    dic = {}
    for nr_idividual in range(nr_individuals):
        
        hexa_list = [token_hex(4) for i in range(nr_of_genes)]
        dic[nr_idividual] = gene_to_neuron(hexa_list: list, nr_of_input, nr_of_actions, nr_of_inner)

    return dic  

def initial_population( nr_individuals: int, nr_of_genes: int, nr_of_input: int, nr_of_actions: int, nr_of_inner: int, world_size: int):
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

