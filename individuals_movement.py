from common_functions import *

import random
import numpy as np
import copy
from tqdm import tqdm
from common_functions import *

def steps_in_generation(movement_count: int, result: dict, world_size_x: int, world_size_y: int):
    '''calculate steps of each generation including factors came from initial neurons
    movement_count = number of stpes of each individuals. default - world_size
    result = dictionary with individuals' parameters
    world_size_x and world_size_y = in this version equals world_size
    '''
    move = 0
    pbar = tqdm(total=movement_count, initial=move)

    while movement_count > move:

        pbar.update(1)
        
        for indiv in result:
            # last position of each individual
            x, y = result[indiv]['position'][-1][0], result[indiv]['position'][-1][1]
            if move < 1:
                calculate_position(result, indiv, x, y, world_size_x, world_size_y)    
            elif move >= 1:
                apply_input(result, indiv)
                calculate_position(result, indiv, x, y, world_size_x, world_size_y)
                
        last_pos_list = {obj:result[obj]['position'][-1] for obj in result}
        prevent_overlap_movement(last_pos_list, result)
        move += 1
    pbar.close()
    return result

################### start ###################
def calculate_position(result, indiv, x, y, world_size_x, world_size_y):
    out_weight = result[indiv]['out']
    position_list = (move(out, out_weight[out]) for out in out_weight)

    if position_list:
        position_list = list(map(sum, zip(*position_list)))
        position_list = make_smaller_(position_list)
        position_list = list(map(sum, zip(*[[x, y]] + [position_list])))

        position_list[0] = normalize_position_if_outside_world(position_list[0], world_size_x)
        position_list[1] = normalize_position_if_outside_world(position_list[1], world_size_y)

        result[indiv]['position'].append(position_list)
        
## TODO check if 'out4' is properly executed 
def move(key, weight):
    factor_1 = np.random.choice(2, 1, p=[weight, 1-weight])
    if 'out0' in key:
        return [0, int(factor_1)]
    elif 'out1' in key:
        return [0, int(-factor_1)]
    elif 'out2' in key:
        return [int(+factor_1), 0]
    elif 'out3' in key:
        return [int(-factor_1), 0]
    elif 'out4' in key:
        x,y = np.random.choice(2, 2)
        return [x,y]
        
def make_smaller_(l):
    '''-2 to -1, 2 to 1'''
    for i_nr, i in enumerate(l):
        if i == -2:
            l[i_nr] = -1
        elif i == 2:
            l[i_nr] = 1
    return l
    
def normalize_position_if_outside_world(position, max_border):
    '''limit position to world border'''
    if position < 0:
        position = 0
    elif position > max_border:
        position = max_border    
    return position

################### stop ###################


################### start ###################

def apply_input(result, nr_of_individual):
    '''apply input regarding position of other individuals'''

    pos = result[nr_of_individual]['position']
    in_keys = result[nr_of_individual]['in']

    result = sum_input_weights(result, nr_of_individual, in_keys, pos)

    edges = result[nr_of_individual]['brain']
    # print(edges)
    # edges = (tuple(edges[i]) for i in edges)
    set_neurons = set(i[1] for i in edges.values())
    remove_mid_with_no_predecessor(edges, set_neurons)

    mid_dic = {}
    populate_dictionary(edges, mid_dic)
    
    sum_weights(mid_dic, in_keys)


    remove_mid_from_dict(mid_dic)
    
    result[nr_of_individual]['brain_after_pruning'] = edges
    result[nr_of_individual]['out'] = mid_dic
      
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

def input_neuron(key, pos, result):
    '''key - input name
        pos - list of individual position
        tot_position - all individual last position '''
    if pos[-2] != pos[-1]:
        x1, y1, x2, y2 = pos[-2][0], pos[-2][1], pos[-1][0], pos[-1][1]
        dx, dy = x2-x1, y2-y1
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

def check_overlap(result, x, y):
    for indiv in result:
        if [x, y] == result[indiv]['position'][-1]:
            factor = 1
        else:
            factor = 0
        return factor

################### stop ###################

################### start ###################
def prevent_overlap_movement(last_pos_list, result):
    '''check if last position of each individual ovrlap with another. If yes then last posotion is switched to last but one. 
    last_pos_list - dictionary of individuals kesy and last position
    result- total info about all individuals'''
    list_of_resuls = []
    for key_1, val_1 in last_pos_list.items():
        last_pos_list_copy = copy.copy(last_pos_list)
        del last_pos_list_copy[key_1]
        list_of_keys = []
        for key_2, val_2 in last_pos_list_copy.items():
            if val_1 == val_2:
                pos2_minus = result[key_2]['position']
                pos1_minus = result[key_1]['position']
                
                if pos2_minus[-1] != pos2_minus[-2]:
                    pos2_minus[-1] = pos2_minus[-2]
                    last_pos_list[key_2] = pos2_minus[-2]
                    prevent_overlap_movement(last_pos_list, result)
                    list_of_resuls.append(2)
                elif pos2_minus[-1]== pos2_minus[-2] and pos1_minus[-1] != pos1_minus[-2]:
                    pos1_minus[-1] = pos1_minus[-2]
                    last_pos_list[key_1] = pos1_minus[-2]
                    prevent_overlap_movement(last_pos_list, result)
                    list_of_resuls.append(1)
                else:
                    list_of_resuls.append([[key_1, key_2]])
                    
################### stop ###################