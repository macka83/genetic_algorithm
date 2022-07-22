import numpy as np
import copy
# def remove_mid_with_no_predecessor(edges):
    # '''remove mid neuron if no predecessor'''
    # set_neurons = set(i[1] for i in edges)
    # for i_nr, i in enumerate(edges):
        # if 'mid' in i[0] and i[0] not in set_neurons:
            # del edges[i_nr]
            # remove_mid_with_no_predecessor(edges)
            
def remove_mid_with_no_predecessor(edges, set_neurons):
    '''remove mid neuron if no predecessor'''
    for val_nr, val in enumerate(edges):
        if 'mid' in val[0] and val[0] not in set_neurons:
            del edges[val_nr]
            remove_mid_with_no_predecessor(edges, set_neurons)

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
        
def populate_dictionary(edges, dic):
    '''generate dictionary where keys are outputs: out, mid and values are distionaries of in , mid and their weights
    edges- dict with edges
    dic- empty dictionary'''
    for item in edges.values():
        if item[1] in dic:
            dic[item[1]].update({item[0]: item[2]})
        else:
            dic[item[1]] = {item[0]: item[2]}