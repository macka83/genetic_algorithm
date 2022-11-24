from tqdm import tqdm
import numpy as np
import copy
import time

def steps_in_generation(world_size, result, world_size_x, world_size_y):
    n = 0
    # pbar = tqdm(total=world_size, initial=n)

    while  world_size>n:

        # pbar.update(1)
        s0 = time.time()
        for indiv in result:
            x, y = result[indiv]['position'][-1][0], result[indiv]['position'][-1][1]
            if n<1:
            
                start_1 = time.time()
                calculate_position(result, indiv, x, y, world_size_x, world_size_y)
                end_1 = time.time()
                
            elif n >= 1:
            
                start_2 = time.time()
                apply_input(result, indiv)
                end_2 = time.time()
                
                start_3 = time.time()
                calculate_position(result, indiv, x, y, world_size_x, world_size_y)
                end_3 = time.time()
                
        # last_pos_list = {obj:result[obj]['position'][-1] for obj in result}
        
            start_4 = time.time()
            # prevent_overlap_movement(last_pos_list, result)
            for prev in sorted(result.keys()): 
                if prev != indiv and result[prev]['position'][-1] == result[indiv]['position'][-1]:
                    result[indiv]['position'][-1] = result[indiv]['position'][-2]
                    
            # result_copy = result.copy() 
            # del result_copy[indiv]
            # if {tuple(result_copy[indiv]['position'][-1]) for indiv in result_copy}&{tuple(result[indiv]['position'][-1])}:
                # result[indiv]['position'][-1] = result[indiv]['position'][-2]
    
            end_4 = time.time()
        n += 1
        e0 = time.time()
    # pbar.close()
    yield result, end_1-start_1, end_2-start_2, end_3-start_3, end_4-start_4, e0-s0
   

   
  
## from 'steps_in_generation'  
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
  
## from 'steps_in_generation'  
def apply_input(result, nr_of_individual):
    '''apply input regarding position of other individuals'''

    pos, in_keys = result[nr_of_individual]['position'], result[nr_of_individual]['in']

    
    start = time.time()
    result = sum_input_weights(result, nr_of_individual, in_keys, pos)
    end = time.time()
    
    edges = result[nr_of_individual]['brain']
    edges = [tuple(edges[i]) for i in edges]
    remove_mid_with_no_predecessor(edges)

    mid_dic = {}
    for item in edges:
        if item[1] in mid_dic:
            mid_dic[item[1]].update({item[0]: item[2]})
        else:
            mid_dic[item[1]] = {item[0]: item[2]}
    start = time.time()
    sum_weights(mid_dic, in_keys)
    end = time.time()
    
    start = time.time()
    remove_mid_from_dict(mid_dic)
    end = time.time()
    
    result[nr_of_individual]['brain_after_pruning'] = edges
    result[nr_of_individual]['out'] = mid_dic
  

 
## from 'steps_in_generation' 
def calculate_position(result, indiv, x, y, world_size_x, world_size_y):

    position_list = (move(out, result[indiv]['out'][out]) for out in result[indiv]['out'])

    if position_list:
        position_list = make_smaller_(list(map(sum, zip(*position_list))))
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

def remove_mid_with_no_predecessor(edges):
    '''remove mid neuron if no predecessor'''
    set_neurons = set(i[1] for i in edges)
    for i_nr, i in enumerate(edges):
        if 'mid' in i[0] and i[0] not in set_neurons:
            del edges[i_nr]
            remove_mid_with_no_predecessor(edges)
            

def check_overlap(result, x, y):
    for indiv in result:
        if [x, y] == result[indiv]['position'][-1]:
            input = 1
        else:
            input = 0
        return input

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

def NormalizeData(data):
    return round((data + 1) / 2, 3)
    

def remove_mid_from_dict(dic):
    mid_list = [i for i in dic if 'mid' in i]
    for mid in mid_list:
        dic.pop(mid)            
       
    
