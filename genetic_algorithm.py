#!/usr/bin/env python
# coding: utf-8

# https://github.com/davidrmiller/biosim4
# 

# In[ ]:


from secrets import token_hex
import secrets
import random
from collections import Counter
from itertools import groupby

import numpy as np
import matplotlib.pyplot as plt


# ### funkcje neuronów

# input

# In[ ]:


# Bliska przeszkoda
# 0in
def close_obstacle(obj_loc_x, obj_loc_y, obj_list):
    select_obst = 0
    # iteruj po liście bez obiektu
    for obj_nr in obj_list:
        if obj_nr['coord'][0] - obj_loc_x == abs(1) and obj_nr['coord'][1] - obj_loc_y == abs(1):
            select_obst += 1
            
    return select_obst/8
       
# Daleka przeszkoda gdy ruch po osi x
# 1in
def distant_obstacle_x_coord(obj_loc_x, obj_list):
    select_obst = 0
    # iteruj po liście bez obiektu
    for obj_nr in obj_list:
        if obj_nr['coord'][0] - obj_loc_x <= abs(5):
            select_obst += 1
            
    return select_obst/121

# Daleka przeszkoda gdy ruch po osi y
# to samo j.w.
# 2in
def distant_obstacle_y_coord(obj_loc_y, obj_list):
    select_obst = 0
    # iteruj po liście bez obiektu
    for obj_nr in obj_list:
        if obj_nr['coord'][0] - obj_loc_y <= abs(5):
            select_obst += 1
            
    return select_obst/121


# output

# przykład jak zastosować neurony executeActions.cpp

def move_x(weight):
    weight_abs = abs(weight)
    x = np.random.choice(2, 1, p=[1-weight_abs, weight_abs])
    if weight < 0:
        x = x * -1
    return x

def move_y(weight):
    weight_abs = abs(weight)
    y = np.random.choice(2, 1, p=[1-weight_abs, weight_abs])
    if weight < 0:
        y = y * -1
    return y

def random_move():
    x,y = np.round(np.random.uniform(low=-1, high=1.0, size=2))
    return x,y
    


# decode hexadecimal

def split_genome(hexval):
    binary = bin(int(hexval, 16))[2:]
    if len(binary) < 32:
        factor = 32 - len(binary)
        binary = '0' * factor + binary
  
    source_type, source_id = binary[0], binary[1:8]
    sink_type, sink_id = binary[8], binary[9:16]
    weight_sign, weight = binary[16], binary[17:]

    gene_value_list = {key:int(val, 2) for key, val in zip(['source_type', 'source_id', 'sink_type', 'sink_id', 'weight_sign', 'weight'],
                                          [source_type, source_id, sink_type, sink_id, weight_sign, weight])}
    
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

    differ_neuron = f'{input_id}{input_type}{output_id}{output_type}'
    
    return input_id, input_type, weight, output_id, output_type, differ_neuron

# def sum_weights_from_duplicated_neurons(gene_translated, l):
    # if gene_translated:
        # for neuron in gene_translated:
            # if neuron.differ_neuron == l.differ_neuron:
                # neuron.weight = neuron.weight + l.weight
            # else:
                # gene_translated.append(l)

    else:
        gene_translated.append(l)
    print('final', [i.differ_neuron for i in gene_translated])

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
#         print()
    return lista          

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
            n_output = f'{i.output_id}{i.output_type}'
            n_input = f'{i.input_id}{i.input_type}'
            dic[nr][i.differ_neuron] = [n_input, n_output, total]
    return dic

def remove_self_loop(dic):
    '''remove neurons if self looped ie: Amid->Bmid, Bmid->Amid'''
    for nr in dic:
        list_of_dup = []
        for key_1 in dic[nr]:
            for key_2 in dic[nr]:
                if key_1 != key_2 and sorted(key_1) == sorted(key_2):
                    list_of_dup.append(key_1)

        for dup in list_of_dup:
            del dic[nr][dup]
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

def analyze_brain(brain):
    for k, v in brain.items():
        if isinstance(v, dict) and len(v) >= 2:
            print('if', k, v)
#             analyze_brain(brain)
        else:
            print('else', k, v)
            
# calculate weight sum

def mid_to_weight(dic):
    '''weight for clear input ie. no nested weights: 0mid:{0in, 1in} or 0mid:{0mid, 0in}'''
    for key in dic:
        if 'mid' in key and isinstance(dic[key], dict):
            if set(i[:2] for i in dic[key]) == set(['in']):
                dic[key] = np.tanh(sum(dic[key].values()))
                mid_to_weight(dic)
                
            elif key in dic[key].keys() and Counter(i[:2] for i in dic[key])['mi'] == 1:
                dic[key] = np.tanh(sum(dic[key].values()))
                mid_to_weight(dic)
                
def out_to_weight(dic):
    '''weight for nested input ie.  0mid:{1mid} or out:{0mid, 0in}'''
    for key in dic:
        if 'mid' in key and isinstance(dic[key], dict):
            for sub_key in dic[key]:
                if 'mid' in sub_key and sub_key != key:
                    dic[key][sub_key] += dic[sub_key]

            dic[key] = np.tanh(sum(dic[key].values()))
            out_to_weight(dic)
                
        elif 'out' in key and isinstance(dic[key], dict):
            for sub_key in dic[key]:
                if 'mid' in sub_key:
                    dic[key][sub_key] += dic[sub_key]
            dic[key] = np.tanh(sum(dic[key].values()))
            out_to_weight(dic)

## preprocessing            
def weight_sum_preprocessing(edges):
    '''"out" and "mid" neurons to key. Values are predecessors neurons ie. "in" or "mid"'''
    d = {}
    for item in edges:
        if item[1] in d:
            d[item[1]].update({item[0]:item[2]})
        else:
            d[item[1]] = {item[0]:item[2]}

    dic = {}
    for i in sorted(d):
        dic[i] = d[i]
    
    return dic