#!/usr/bin/env python
# coding: utf-8

# https://github.com/davidrmiller/biosim4
# 

from secrets import token_hex
import random
from itertools import groupby
from itertools import tee
import numpy as np
import copy


# ### funkcje neuronów

# input
# # Bliska przeszkoda

def check_overlap(result, x, y):
    for indiv in result:
        if [x, y] == result[indiv]['position'][-1]:
            input = 1
        else:
            input = 0
        return input

def input_neuron(key, pos, result):
    '''key - input name
        pos - list of individual position
        tot_position - all individual last position '''
    if pos[-2] != pos[-1]:
        x1, y1, x2, y2 = pos[-2][0], pos[-2][1], pos[-1][0], pos[-1][1]
        dx = x2-x1
        dy = y2-y1
        x3, y3 = x2+dx, y2+dy
        
        if 'in0' in key:
            #close obstacle
            # return 0 or 1
            return key, check_overlap(result, x3, y3)
        elif 'in1' in key:
            #distant obstacle (5 steps forward
            # return between 0 and 1
            for i in range(5):
                if dx != 0:
                    dx += 1
                if dy != 0:
                    dy += 1
                factor = check_overlap(result, x2+dx, y2+dy)
                if factor == 0:
                    return key, 0
                    break
                else:
                    return key, i/5
                    break
                    
    else:
        return key, 0
# output
# updated output neuron
def move(key, weight):
    factor_1 = np.random.choice(2, 1, p=[1-weight, weight])
    if 'out0' in key:
        return [0, int(factor_1)]
    elif 'out1' in key:
        return [0, int(-factor_1)]
    elif 'out2' in key:
        return [int(+factor_1), 0]
    elif 'out3' in key:
        return [int(-factor_1), 0]
    elif 'out4' in key:
        factor_2 = np.random.choice(2, 1, p=[1-weight, weight])
        return [int(-factor_1), int(-factor_2)]

# decode hexadecimal

def split_genome(hexval):
    binary = bin(int(hexval, 16))[2:]
    if len(binary) < 32:
        factor = 32 - len(binary)
        binary = '0' * factor + binary

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

def generate_initial_genomes_for_population(nr_individuals, nr_of_genes, nr_of_input, nr_of_actions, nr_of_inner):
    dic = {}
    for nr_idividual in range(nr_individuals):
        gene_translated = []
        hexa_list = [token_hex(4) for i in range(nr_of_genes)]
        for l, hex_id in enumerate(hexa_list):

            gen_component = split_genome(hex_id)
            
            input_id, input_type, weight, output_id, output_type, differ_neuron = get_neurons_body(gen_component, nr_of_input, nr_of_inner, nr_of_actions)
            
            l = Neuron(hex_id, input_id, input_type, weight, output_id, output_type, differ_neuron)
            gene_translated.append(l)
#             sum_weights_from_duplicated_neurons(gene_translated, l)

        dic[nr_idividual] = gene_translated

    return dic          

class Creature():
    def __init__(self, brain, x, y):
        self.brain = brain
        self.x = x
        self.y = y

class Neuron():
    def __init__(self, hex_id, input_id, input_type, weight, 
                 output_id, output_type, differ_neuron):
        self.hex_id = hex_id
        self.input_id = input_id
        self.input_type = input_type
        self.weight = weight
        self.output_id = output_id
        self.output_type = output_type
        self.differ_neuron = differ_neuron
        
# brain generator

def sum_duplicated_neurons(res):
    '''sum duplicatd neurons and return bunch neurons dictionary'''
    dic = {}
    for nr in res:
        dic[nr] = {}
        for i in res[nr]:
            total = dic.get(i.differ_neuron, 0) + i.weight
            n_output = f'{i.output_type}{i.output_id}'
            n_input = f'{i.input_type}{i.input_id}'
            dic[nr][i.differ_neuron] = [n_input, n_output, total]
    return dic

def remove_self_loop(dic):
    '''remove randomlypicked self looped ie: Amid->Bmid or Bmid->Amid'''
    for nr in dic:
        list_of_dup = []
        for key_1 in dic[nr]:
            for key_2 in dic[nr]:
                if key_1 != key_2 and sorted(key_1) == sorted(key_2):
                    list_of_dup.append(sorted([key_1, key_2]))
                    
        list_of_dup.sort()
        list_of_dup = list(list_of_dup for list_of_dup,_ in groupby(list_of_dup))
        for i in list_of_dup:
            rand_int = random.randint(0, 1)
            del dic[nr][i[rand_int]]

    return dic

def generate_brain_output_dictionary(edges):
    '''generate list of outputs dictionary to store 'mid' and 'in' neurons'''
    brain_output_template = {}
    for i in edges:
        if 'out' in edges[i][1]:
            brain_output_template.update({edges[i][1]:{}})
    return brain_output_template

def mid_neuron(brain, edges):
    for key in brain:
        if 'out' in key:
            for pair in edges:
                item = edges[pair]
                if key == item[1] and 'mid' in item[0] and key != item[0]:
                    brain[key].update({item[0]: {'w':item[2]}})
                    mid_neuron(brain[key], edges)
                elif key == item[1] and 'mid' in item[0] and key == item[0]:
                    brain[key].update({item[0]:{'w':item[2]}})
                elif key == item[1] and 'mid' not in item[0]:
                    brain[key].update({item[0]:{'w':item[2]}})
           
# calculate weight sum

def remove_mid_from_dict(dic):
    mid_list = [i for i in dic if 'mid' in i]
    for mid in mid_list:
        dic.pop(mid)            
            
## preprocessing            

def remove_mid_with_no_predecessor(edges):
    '''remove mid neuron if no predecessor'''
    set_neurons = set(i[1] for i in edges)
    for i_nr, i in enumerate(edges):
        if 'mid' in i[0] and i[0] not in set_neurons:
            del edges[i_nr]
            remove_mid_with_no_predecessor(edges)

## calculate paths in-mid-out and weights
def NormalizeData(data):
    return round((data + 1) / 2, 3)

def sum_weights(dic, input_list):
    '''input: dic - dictionary of 'mid' and 'out' neurons with predecessors
        ex.{0: {'out1': {'mid0': -2.8534106516099498, 'mid1': -0.6730352510300626},
                'mid1': {'in2': -1.9940790477643828, 'mid0': -3.1373721959407903,'mid1': -1.880543262627804},
                'out0': {'in2': 0.16151381046848773},
                'mid0': {'in1': 2.8191057530901875}},
                0 = nr of individual,
       filter_list - list of inputs ex. ['in0', 'in2', 'in1']'''
    for key in dic:
        if 'mid' in key and isinstance(dic[key], dict):
            mid_to_update = set(dic[key]).difference(set(input_list + [key]))
            k = 0
            for mid_key in mid_to_update:
                if isinstance(dic[mid_key], float):
                    dic[key][mid_key] = NormalizeData(np.tanh(sum([dic[key][mid_key], dic[mid_key]])))
                    k+=1
            if k == len(mid_to_update):
                dic[key] = np.tanh(sum(dic[key].values()))
                sum_weights(dic, input_list)
                
        elif 'out' in key and isinstance(dic[key], dict):
            for mid_key in dic[key]:
                if 'mid' in mid_key and isinstance(dic[mid_key], float):
                    dic[key][mid_key] = NormalizeData(np.tanh(sum([dic[key][mid_key], dic[mid_key]])))

            dic[key] = NormalizeData(np.tanh(sum(dic[key].values())))   

def calculate_individual_output_weights(individuals):
    dic = {}
    ## sum duplicates
    individuals_sum_dup = sum_duplicated_neurons(individuals)
    individuals_sum_dup_no_self_loop = remove_self_loop(individuals_sum_dup)
    
    for individual in individuals_sum_dup_no_self_loop:

        ## preprocessing
        edges = individuals_sum_dup_no_self_loop[individual]
        edges = [tuple(edges[i]) for i in edges]
        remove_mid_with_no_predecessor(edges)

        init_list = list(set([i[0] for i in edges if 'in' in i[0]]))

        mid_dic = {}

        for item in edges:
            if item[1] in mid_dic:
                mid_dic[item[1]].update({item[0]: item[2]})
            else:
                mid_dic[item[1]] = {item[0]: item[2]}
        
        if mid_dic:
            sum_weights(mid_dic, init_list)
            remove_mid_from_dict(mid_dic)
        
            dic[individual] = {}
            dic[individual]['out'] = {}
            dic[individual]['brain'] = {}
            dic[individual]['in'] = {}
            dic[individual]['out'] = mid_dic
            dic[individual]['brain'] = individuals_sum_dup_no_self_loop[individual]
            dic[individual]['in'] = init_list

        
    return dic

def normalize_position_if_outside_world(position, max_border):
    '''limit position to world border'''
    if position < 0:
        position = 0
    elif position > max_border:
        position = max_border

        
    return position

def make_smaller_(l):
    '''-2 to -1, 2 to 1'''
    for i_nr, i in enumerate(l):
        if i == -2:
            l[i_nr] = -1
        elif i == 2:
            l[i_nr] = 1
    return l
    
def sum_input_weights(result, nr_of_individual, in_keys, pos):
    '''sum calculated input based on environment with neurons containing input
    result- list of creatures
    nr_of_individual- number individual
    in_keys- list of inputs id
    pos- list of position of individual'''
    result_copy = copy.copy(result)
    del result_copy[nr_of_individual]
    
    for key in in_keys:
        in_weight = input_neuron(key, pos, result_copy)
        for neuron in result[nr_of_individual]['brain']:
            # print(result[nr_of_individual]['position'])
            # print(in_weight)
            if in_weight[0] == result[nr_of_individual]['brain'][neuron][0]:
                result[nr_of_individual]['brain'][neuron][2] += in_weight[1]
                
    return result 
    
def apply_input(result, nr_of_individual):
    '''apply input regarding position of other individuals'''

    pos = result[nr_of_individual]['position']
    in_keys = result[nr_of_individual]['in']

    result = sum_input_weights(result, nr_of_individual, in_keys, pos)

    edges = result[nr_of_individual]['brain']
    edges = [tuple(edges[i]) for i in edges]
    remove_mid_with_no_predecessor(edges)

    mid_dic = {}
    for item in edges:
        if item[1] in mid_dic:
            mid_dic[item[1]].update({item[0]: item[2]})
        else:
            mid_dic[item[1]] = {item[0]: item[2]}
    
    sum_weights(mid_dic, in_keys)


    remove_mid_from_dict(mid_dic)
    
    result[nr_of_individual]['brain_after_pruning'] = edges
    result[nr_of_individual]['out'] = mid_dic
    

def prevent_overlap_movement(last_pos_list, result):
    '''check if last position of each individual ovrlap with another. If yes then last posotion is switched to last but one. 
    last_pos_list - dictionary of individuals kesy and last position
    result- total info about all individuals'''
    for key_1, val_1 in last_pos_list.items():
        last_pos_list_copy = copy.copy(last_pos_list)
        del last_pos_list_copy[key_1]
        for key_2, val_2 in last_pos_list_copy.items():
            if val_1 == val_2:
                pos_minus = result[key_2]['position']

                if pos_minus[-1] != pos_minus[-2]:
                    pos_minus[-1] = pos_minus[-2]
                    last_pos_list[key_2] = pos_minus[-2]
                    prevent_overlap_movement(last_pos_list, result)
                
def calculate_position(result, indiv, x, y, world_size_x, world_size_y):
    position_list = []
    for out in result[indiv]['out']:
        new_pos = move(out, result[indiv]['out'][out])
        position_list.append(new_pos)
    
    if position_list:
        position_list = list(map(sum, zip(*position_list)))
        position_list = make_smaller_(position_list)
        position_list = list(map(sum, zip(*[[x, y]] + [position_list])))

        position_list[0] = normalize_position_if_outside_world(position_list[0], world_size_x)
        position_list[1] = normalize_position_if_outside_world(position_list[1], world_size_y)

        result[indiv]['position'].append(position_list)