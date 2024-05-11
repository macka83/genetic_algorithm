#!/usr/bin/env python
# coding: utf-8

# https://github.com/davidrmiller/biosim4
#

from secrets import token_hex
import random
from itertools import combinations, product
import numpy as np
import copy
from tqdm import tqdm


class BrainInitiation:
    def __init__(
        self, nr_of_genes: int, nr_of_input: int, nr_of_actions: int, nr_of_inner: int
    ):
        """
        Initialize the Brain class with specified parameters.

        Args:
            nr_of_genes (int): Number of genes.
            nr_of_input (int): Number of input neurons.
            nr_of_actions (int): Number of output actions.
            nr_of_inner (int): Number of inner neurons.
        """
        self.nr_of_genes = nr_of_genes
        self.nr_of_input = nr_of_input
        self.nr_of_actions = nr_of_actions
        self.nr_of_inner = nr_of_inner

    def generate_initial_genomes_for_population(self) -> list:
        """
        Generate initial genomes for the population.

        Returns:
            list: List of translated neurons.
        """
        hexa_list = [token_hex(4) for _ in range(self.nr_of_genes)]
        return self.gene_to_neuron(hexa_list)

    def gene_to_neuron(self, hexa_list: list) -> list:
        """
        Translate genes to neurons.

        Args:
            hexa_list (list): List of hexadecimal gene IDs.

        Returns:
            list: List of translated neurons.
        """
        gene_translated = []
        for hexval in hexa_list:
            gen_component = self.split_genome(hexval)
            neuron = self.get_neurons_body(hexval, gen_component)
            gene_translated.append(neuron)
        return gene_translated

    def split_genome(self, hexval: str) -> dict:
        """
        Split a hexadecimal genome into components.

        Args:
            hexval (str): Hexadecimal value.

        Returns:
            dict: Dictionary containing gene components.
        """
        binary = self.hexval_to_bin(hexval)

        bin_rep = {
            "source_type": binary[0],
            "source_id": binary[1:8],
            "sink_type": binary[8],
            "sink_id": binary[9:16],
            "weight_sign": binary[16],
            "weight": binary[17:],
        }

        gene_value_list = {key: int(val, 2) for key, val in bin_rep.items()}
        return gene_value_list

    def hexval_to_bin(self, gene: str) -> str:
        """
        Convert a hexadecimal gene to binary.

        Args:
            gene (str): Hexadecimal gene value.

        Returns:
            str: Binary representation.
        """
        binary = bin(int(gene, 16))[2:]
        if len(binary) <= 32:
            factor = 32 - len(binary)
            binary = "0" * factor + binary
        return binary

    def get_neurons_body(self, hexval: str, gen_component: dict) -> tuple:
        """
        Create neuron information based on gene components.

        Args:
            gen_component (dict): Dictionary containing gene components.

        Returns:
            tuple: Neuron information (input_id, input_type, weight, output_id, output_type, differ_neuron).
        """
        input_id = gen_component["source_id"] % (
            self.nr_of_input if gen_component["source_type"] == 0 else self.nr_of_inner
        )
        input_type = "in" if gen_component["source_type"] == 0 else "mid"

        output_id = gen_component["sink_id"] % (
            self.nr_of_inner if gen_component["sink_type"] == 0 else self.nr_of_actions
        )
        output_type = "mid" if gen_component["sink_type"] == 0 else "out"

        weight_sign = -1 if gen_component["weight_sign"] == 0 else 1
        weight = weight_sign * (gen_component["weight"] / 8191.25)

        differ_neuron = f"{input_type}{input_id}{output_type}{output_id}"

        return (
            hexval,
            f"{input_type}{input_id}",
            weight,
            f"{output_type}{output_id}",
            differ_neuron,
        )


class Population:
    def __init__(self, world_size: int, nr_individuals: int, brain: dict):
        self.world_size = world_size
        self.nr_individuals = nr_individuals
        self.brain = brain

    def generate_random_coords(self) -> any:
        """
        Generates random coordinates for individuals within the specified world size.

        Args:
            self.world_size (int): The size of the world (assumed to be square).

        Returns:
            np.ndarray: An array of randomly selected coordinates for individuals.
        """
        coord_full_list = np.array(list(product(range(self.world_size), repeat=2)))
        ind = np.random.choice(len(coord_full_list), self.nr_individuals, replace=False)
        pos = coord_full_list[ind]
        return pos

    def assign_position(self, pos: list) -> None:
        """
        Assigns positions to individuals

        Args:
            brain (dict): A dictionary containing individual information.
            pos (list): List of positions for individuals.

        Returns:
            None: The function modifies the 'brain' dictionary in-place.
        """

        for indiv in self.brain:
            self.brain[indiv]["position"] = [list(pos[indiv])]

    def remove_outputless_brains(self) -> None:
        """
        removes outputless brain.

        Args:
            brain (dict): A dictionary containing individual information.

        Returns:
            None: The function modifies the 'brain' dictionary in-place.
        """
        indiv_to_del = (indiv for indiv in self.brain if not self.brain[indiv]["out"])
        for key in indiv_to_del:
            del self.brain[key]

    def clear_malfunction_brain_and_assign_position(self):
        pos = self.generate_random_coords()
        self.assign_position(pos)
        self.remove_outputless_brains()


class CalculateWeights:

    def __init__(self, brain: list):
        self.brain = brain

    def normalize_data(self, data: float) -> float:
        return round((data + 1) / 2, 3)

    def sum_duplicated_neurons(self) -> dict:
        cumulative_sum_dict = {}

        # Iterate through the list of sublists
        for sublist in self.brain:
            differ_neuron = sublist[-1]
            weight = sublist[2]

            # If the last item is not already in the dictionary, create a new entry
            if differ_neuron not in cumulative_sum_dict:
                cumulative_sum_dict[differ_neuron] = [
                    f"{sublist[1]}",
                    f"{sublist[-2]}",
                    weight,
                ]
            else:
                # Update the cumulative sum for the existing last item
                cumulative_sum_dict[differ_neuron][2] += weight

        return cumulative_sum_dict

    def remove_self_loop(self, brain_to_dict: dict) -> dict:
        """
        Remove randomly picked self-looped edges, e.g., Amid->Bmid or Bmid->Amid.
        """
        keys = list(brain_to_dict.keys())  # Extract keys once
        # print(keys)
        key_combinat = combinations(keys, 2)
        for key_pair in key_combinat:
            if key_pair[0] != key_pair[1] and sorted(key_pair[0]) == sorted(
                key_pair[1]
            ):
                rand_int = random.randint(0, 1)
                del brain_to_dict[key_pair[rand_int]]

        return brain_to_dict

    def sum_weights(self, dic: dict, input_list: list) -> None:
        """input: dic - dictionary of 'mid' and 'out' neurons with predecessors
         ex.{0: {'out1': {'mid0': -2.8534106516099498, 'mid1': -0.6730352510300626},
                 'mid1': {'in2': -1.9940790477643828, 'mid0': -3.1373721959407903,'mid1': -1.880543262627804},
                 'out0': {'in2': 0.16151381046848773},
                 'mid0': {'in1': 2.8191057530901875}},
                 0 = nr of individual,
        filter_list - list of inputs ex. ['in0', 'in2', 'in1']"""
        for key in dic:
            if "mid" in key and isinstance(dic[key], dict):
                mid_to_update = set(dic[key]).difference(set(input_list + [key]))
                k = 0
                for mid_key in mid_to_update:
                    if isinstance(dic[mid_key], float):
                        dic[key][mid_key] = self.normalize_data(
                            np.tanh(sum([dic[key][mid_key], dic[mid_key]]))
                        )
                        k += 1
                if k == len(mid_to_update):
                    dic[key] = np.tanh(sum(dic[key].values()))
                    self.sum_weights(dic, input_list)

            elif "out" in key and isinstance(dic[key], dict):
                for mid_key in dic[key]:
                    if "mid" in mid_key and isinstance(dic[mid_key], float):
                        dic[key][mid_key] = self.normalize_data(
                            np.tanh(sum([dic[key][mid_key], dic[mid_key]]))
                        )

                dic[key] = self.normalize_data(np.tanh(sum(dic[key].values())))

    def remove_mid_from_dict(self, dic: dict) -> None:
        mid_list = [i for i in dic if "mid" in i]
        for mid in mid_list:
            dic.pop(mid)

    def remove_mid_with_no_predecessor(self, edges: list) -> None:
        """remove mid neuron if no predecessor"""
        set_neurons = set(i[1] for i in edges)
        for i_nr, i in enumerate(edges):
            if "mid" in i[0] and i[0] not in set_neurons:
                del edges[i_nr]
                self.remove_mid_with_no_predecessor(edges)

    def calculate_individual_output_weights(self) -> dict:
        dic = {}

        unique_neurons = self.sum_duplicated_neurons()
        no_selfloop_neurons = self.remove_self_loop(unique_neurons)

        edges = [tuple(no_selfloop_neurons[i]) for i in no_selfloop_neurons]
        self.remove_mid_with_no_predecessor(edges)

        init_list = list(set([i[0] for i in edges if "in" in i[0]]))

        mid_dic = {}

        for item in edges:
            if item[1] in mid_dic:
                mid_dic[item[1]].update({item[0]: item[2]})
            else:
                mid_dic[item[1]] = {item[0]: item[2]}

        if mid_dic:
            self.sum_weights(mid_dic, init_list)
            self.remove_mid_from_dict(mid_dic)

            dic = {
                "out": mid_dic,
                "brain": no_selfloop_neurons,
                "in": init_list,
            }
        return dic


class PopulationMovement(CalculateWeights):

    def __init__(self, world_size, population):
        self.world_size = world_size
        self.population = population

    def steps_in_generation(self) -> dict:
        """
        Simulates steps in the generation process.

        Returns:
            dict: Updated population data after simulation.
        """
        n = 0
        pbar = tqdm(total=self.world_size, initial=n)

        while self.world_size > n:
            pbar.update(1)

            for indiv_nr in self.population:
                last_x_position, last_y_position = (
                    self.population[indiv_nr]["position"][-1][0],
                    self.population[indiv_nr]["position"][-1][1],
                )
                if n < 1:
                    self.calculate_position(
                        self, indiv_nr, last_x_position, last_y_position
                    )
                elif n >= 1:
                    self.apply_input(self, indiv_nr)
                    self.calculate_position(
                        self, indiv_nr, last_x_position, last_y_position
                    )

            last_pos_list = {
                obj: self.population[obj]["position"][-1] for obj in self.population
            }
            self.prevent_overlap_movement(self, last_pos_list)
            n += 1

        pbar.close()
        return self.brain

    def calculate_position(self, indiv_nr: int, x: int, y: int) -> None:
        """
        Calculates the position for an individual based on its output weights.

        Args:
            indiv_nr (int): Individual number.
            x (int): X-coordinate.
            y (int): Y-coordinate.
        """
        out_weight = self.population[indiv_nr]["out"]
        position_list = [self.move(out, out_weight[out]) for out in out_weight]

        if position_list:
            # Sum the coordinates of all positions
            position_list = list(map(sum, zip(*position_list)))

            # Make the position smaller (if needed)
            position_list = self.make_smaller(position_list)

            # Add the current (x, y) coordinates to the position list
            position_list = list(map(sum, zip(*[[x, y]] + [position_list])))

            # Normalize the coordinates if they are outside the world boundaries
            for coord in range(2):
                position_list[coord] = self.normalize_position_if_outside_world(
                    position_list[coord], self.world_size
                )

            # Append the calculated position to the result
            result[indiv_nr]["position"].append(position_list)

    def move(self, key: str, weight: float) -> tuple:
        factor_1 = np.random.choice(2, 1, p=[weight, 1 - weight])
        if "out0" in key:
            return (0, int(factor_1))
        elif "out1" in key:
            return (0, int(-factor_1))
        elif "out2" in key:
            return (int(+factor_1), 0)
        elif "out3" in key:
            return (int(-factor_1), 0)
        elif "out4" in key:
            x, y = np.random.choice(2, 2)
            return (x, y)

    def make_smaller(self, position_list: list) -> list:
        """Convert -2 to -1 and 2 to 1."""
        return [1 if i >= 2 else -1 if i <= -2 else i for i in position_list]

    def normalize_position_if_outside_world(self, position: list) -> list:
        """limit position to world border"""
        if position < 0:
            position = 0
        elif position > self.world_size:
            position = self.world_size

        return position

    def apply_input(self, indiv_nr: int) -> None:
        """apply input regarding position of other individuals"""

        pos = self.brain[indiv_nr]["position"]
        in_keys = self.brain[indiv_nr]["in"]

        self.brain = self.sum_input_weights(self.brain, indiv_nr, in_keys, pos)

        edges = self.brain[indiv_nr]["brain"]
        edges = [tuple(edges[i]) for i in edges]
        self.remove_mid_with_no_predecessor(edges)

        mid_dic = {}
        for item in edges:
            if item[1] in mid_dic:
                mid_dic[item[1]].update({item[0]: item[2]})
            else:
                mid_dic[item[1]] = {item[0]: item[2]}

        self.sum_weights(self, mid_dic, in_keys)
        self.remove_mid_from_dict(self, mid_dic)

        result[indiv_nr]["brain_after_pruning"] = edges
        result[indiv_nr]["out"] = mid_dic

    def prevent_overlap_movement(self, last_pos_dict: dict, result: dict) -> None:
        """check if last position of each individual ovrlap with another. If yes then last posotion is switched to last but one.
        last_pos_dict - dictionary of individuals keys and last position
        result- total info about all individuals"""
        list_of_resuls = []
        for key_1, val_1 in last_pos_dict.items():
            last_pos_list_copy = copy.copy(last_pos_dict)
            del last_pos_list_copy[key_1]

            for key_2, val_2 in last_pos_list_copy.items():
                if val_1 == val_2:
                    pos1_minus, pos2_minus = (
                        result[key_1]["position"],
                        result[key_2]["position"],
                    )

                    if pos2_minus[-1] != pos2_minus[-2]:
                        pos2_minus[-1] = pos2_minus[-2]
                        last_pos_dict[key_2] = pos2_minus[-2]
                        self.prevent_overlap_movement(last_pos_dict, result)
                        list_of_resuls.append(2)
                    elif (
                        pos2_minus[-1] == pos2_minus[-2]
                        and pos1_minus[-1] != pos1_minus[-2]
                    ):
                        pos1_minus[-1] = pos1_minus[-2]
                        last_pos_dict[key_1] = pos1_minus[-2]
                        self.prevent_overlap_movement(last_pos_dict, result)
                        list_of_resuls.append(1)
                    else:
                        list_of_resuls.append([[key_1, key_2]])

    def sum_weights(self, dic: dict, input_list: list) -> None:
        """input: dic - dictionary of 'mid' and 'out' neurons with predecessors
         ex.{0: {'out1': {'mid0': -2.8534106516099498, 'mid1': -0.6730352510300626},
                 'mid1': {'in2': -1.9940790477643828, 'mid0': -3.1373721959407903,'mid1': -1.880543262627804},
                 'out0': {'in2': 0.16151381046848773},
                 'mid0': {'in1': 2.8191057530901875}},
                 0 = nr of individual,
        filter_list - list of inputs ex. ['in0', 'in2', 'in1']"""
        for key in dic:
            if "mid" in key and isinstance(dic[key], dict):
                mid_to_update = set(dic[key]).difference(set(input_list + [key]))
                k = 0
                for mid_key in mid_to_update:
                    if isinstance(dic[mid_key], float):
                        dic[key][mid_key] = NormalizeData(
                            np.tanh(sum([dic[key][mid_key], dic[mid_key]]))
                        )
                        k += 1
                if k == len(mid_to_update):
                    dic[key] = np.tanh(sum(dic[key].values()))
                    self.sum_weights(dic, input_list)

            elif "out" in key and isinstance(dic[key], dict):
                for mid_key in dic[key]:
                    if "mid" in mid_key and isinstance(dic[mid_key], float):
                        dic[key][mid_key] = normalize_data(
                            np.tanh(sum([dic[key][mid_key], dic[mid_key]]))
                        )

                dic[key] = self.normalize_data(np.tanh(sum(dic[key].values())))

    def remove_mid_from_dict(self, dic: dict) -> None:
        mid_list = [i for i in dic if "mid" in i]
        for mid in mid_list:
            dic.pop(mid)


def check_overlap(result: dict, x: int, y: int) -> int:
    return any([x, y] == indiv_data["position"][-1] for indiv_data in result.values())


def input_neuron(key: str, pos: str, result: dict):
    """key - input name
    pos - list of individual position
    tot_position - all individual last position"""
    if pos[-2] != pos[-1]:
        x1, y1, x2, y2 = pos[-2][0], pos[-2][1], pos[-1][0], pos[-1][1]
        dx = x2 - x1
        dy = y2 - y1
        x3, y3 = x2 + dx, y2 + dy

        if "in0" in key:
            # close obstacle
            # return 0 or 1
            return key, check_overlap(result, x3, y3)
        elif "in1" in key:
            # distant obstacle (5 steps forward
            # return between 0 and 1
            for i in range(5):
                if dx != 0:
                    dx += 1
                if dy != 0:
                    dy += 1
                factor = check_overlap(result, x2 + dx, y2 + dy)
                if factor == 0:
                    return key, 0
                else:
                    return key, i / 5

    else:
        return key, 0


## TODO check if 'out4' is properly executed
def move(key: str, weight: float) -> tuple:
    factor_1 = np.random.choice(2, 1, p=[weight, 1 - weight])
    if "out0" in key:
        return (0, int(factor_1))
    elif "out1" in key:
        return (0, int(-factor_1))
    elif "out2" in key:
        return (int(+factor_1), 0)
    elif "out3" in key:
        return (int(-factor_1), 0)
    elif "out4" in key:
        x, y = np.random.choice(2, 2)
        return (x, y)


def generate_brain_output_dictionary(edges):
    """generate list of outputs dictionary to store 'mid' and 'in' neurons"""
    brain_output_template = {}
    for i in edges:
        if "out" in edges[i][1]:
            brain_output_template.update({edges[i][1]: {}})
    return brain_output_template


def mid_neuron(brain, edges):
    for key in brain:
        if "out" in key:
            for pair in edges:
                item = edges[pair]
                if key == item[1] and "mid" in item[0] and key != item[0]:
                    brain[key].update({item[0]: {"w": item[2]}})
                    mid_neuron(brain[key], edges)
                elif key == item[1] and "mid" in item[0] and key == item[0]:
                    brain[key].update({item[0]: {"w": item[2]}})
                elif key == item[1] and "mid" not in item[0]:
                    brain[key].update({item[0]: {"w": item[2]}})


# calculate weight sum


def remove_mid_from_dict(dic):
    mid_list = [i for i in dic if "mid" in i]
    for mid in mid_list:
        dic.pop(mid)


## preprocessing
## from 'steps_in_generation'
def remove_mid_with_no_predecessor(edges):
    """remove mid neuron if no predecessor"""
    set_neurons = set(i[1] for i in edges)
    for i_nr, i in enumerate(edges):
        if "mid" in i[0] and i[0] not in set_neurons:
            del edges[i_nr]
            remove_mid_with_no_predecessor(edges)


## calculate paths in-mid-out and weights
def NormalizeData(data):
    return round((data + 1) / 2, 3)


## from 'steps_in_generation'
def sum_weights(dic, input_list):
    """input: dic - dictionary of 'mid' and 'out' neurons with predecessors
     ex.{0: {'out1': {'mid0': -2.8534106516099498, 'mid1': -0.6730352510300626},
             'mid1': {'in2': -1.9940790477643828, 'mid0': -3.1373721959407903,'mid1': -1.880543262627804},
             'out0': {'in2': 0.16151381046848773},
             'mid0': {'in1': 2.8191057530901875}},
             0 = nr of individual,
    filter_list - list of inputs ex. ['in0', 'in2', 'in1']"""
    for key in dic:
        if "mid" in key and isinstance(dic[key], dict):
            mid_to_update = set(dic[key]).difference(set(input_list + [key]))
            k = 0
            for mid_key in mid_to_update:
                if isinstance(dic[mid_key], float):
                    dic[key][mid_key] = NormalizeData(
                        np.tanh(sum([dic[key][mid_key], dic[mid_key]]))
                    )
                    k += 1
            if k == len(mid_to_update):
                dic[key] = np.tanh(sum(dic[key].values()))
                sum_weights(dic, input_list)

        elif "out" in key and isinstance(dic[key], dict):
            for mid_key in dic[key]:
                if "mid" in mid_key and isinstance(dic[mid_key], float):
                    dic[key][mid_key] = NormalizeData(
                        np.tanh(sum([dic[key][mid_key], dic[mid_key]]))
                    )

            dic[key] = NormalizeData(np.tanh(sum(dic[key].values())))


def normalize_position_if_outside_world(position, max_border):
    """limit position to world border"""
    if position < 0:
        position = 0
    elif position > max_border:
        position = max_border

    return position


def make_smaller(l):
    """Convert -2 to -1 and 2 to 1."""
    return [1 if i == 2 else -1 if i == -2 else i for i in l]


## from 'steps_in_generation'
def sum_input_weights(result, nr_of_individual, in_keys, pos):
    """sum calculated input based on environment with neurons containing input
    result- list of creatures
    nr_of_individual- number individual
    in_keys- list of inputs id
    pos- list of position of individual"""
    result_copy = copy.copy(result)
    del result_copy[nr_of_individual]

    for key in in_keys:
        in_weight = input_neuron(key, pos, result_copy)
        for neuron in result[nr_of_individual]["brain"]:
            # print(result[nr_of_individual]['position'])
            # print(in_weight)
            if in_weight[0] == result[nr_of_individual]["brain"][neuron][0]:
                result[nr_of_individual]["brain"][neuron][2] += in_weight[1]

    return result


## from 'steps_in_generation'
def apply_input(result, nr_of_individual):
    """apply input regarding position of other individuals"""

    pos = result[nr_of_individual]["position"]
    in_keys = result[nr_of_individual]["in"]

    result = sum_input_weights(result, nr_of_individual, in_keys, pos)

    edges = result[nr_of_individual]["brain"]
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

    result[nr_of_individual]["brain_after_pruning"] = edges
    result[nr_of_individual]["out"] = mid_dic


## from 'steps_in_generation'
def prevent_overlap_movement(last_pos_list, result):
    """check if last position of each individual ovrlap with another. If yes then last posotion is switched to last but one.
    last_pos_list - dictionary of individuals kesy and last position
    result- total info about all individuals"""
    list_of_resuls = []
    for key_1, val_1 in last_pos_list.items():
        last_pos_list_copy = copy.copy(last_pos_list)
        del last_pos_list_copy[key_1]
        list_of_keys = []
        for key_2, val_2 in last_pos_list_copy.items():
            if val_1 == val_2:
                pos2_minus = result[key_2]["position"]
                pos1_minus = result[key_1]["position"]

                if pos2_minus[-1] != pos2_minus[-2]:
                    pos2_minus[-1] = pos2_minus[-2]
                    last_pos_list[key_2] = pos2_minus[-2]
                    prevent_overlap_movement(last_pos_list, result)
                    list_of_resuls.append(2)
                elif (
                    pos2_minus[-1] == pos2_minus[-2]
                    and pos1_minus[-1] != pos1_minus[-2]
                ):
                    pos1_minus[-1] = pos1_minus[-2]
                    last_pos_list[key_1] = pos1_minus[-2]
                    prevent_overlap_movement(last_pos_list, result)
                    list_of_resuls.append(1)
                else:
                    list_of_resuls.append([[key_1, key_2]])


## from 'steps_in_generation'
def calculate_position(result, indiv, x, y, world_size_x, world_size_y):
    out_weight = result[indiv]["out"]
    position_list = (move(out, out_weight[out]) for out in out_weight)

    if position_list:
        position_list = list(map(sum, zip(*position_list)))
        position_list = make_smaller_(position_list)
        position_list = list(map(sum, zip(*[[x, y]] + [position_list])))

        position_list[0] = normalize_position_if_outside_world(
            position_list[0], world_size_x
        )
        position_list[1] = normalize_position_if_outside_world(
            position_list[1], world_size_y
        )

        result[indiv]["position"].append(position_list)


## mutation


def genome_mutation(genome: dict) -> None:
    for gene_nr, gene in enumerate(genome):
        binary_gene = hexval_to_bin(gene)
        binary_mutated = mutation(binary_gene, weight=0.09)
        if binary_mutated != binary_gene:
            genome[gene_nr] = hex(int(binary_mutated, 2))[2:]


def mutation(binary_gene: str, weight=0.001):
    """make punctual mutation on gene with given weight
    binary_gene - binary string - '010101011100'
    weight - float between 0-1
    """
    ind = np.random.randint(len(binary_gene))
    if int(binary_gene[ind]) == 1:
        mutant = np.random.choice(2, 1, p=[weight, 1 - weight])
    else:
        mutant = np.random.choice(2, 1, p=[1 - weight, weight])

    binary_to_ints = [i for i in list(binary_gene)]
    binary_to_ints[ind] = str(mutant[0])

    return "".join(binary_to_ints)


## visualisation


def hex_to_rgb(hex_value: str):
    #     h = hex_value.lstrip('#')
    return tuple(int(hex_value[i : i + 2], 16) / 255.0 for i in (0, 2, 4))


def generate_dictionary_of_coords(result: dict, list_length: int, dic_color: dict):
    """generate list of x and y dictionary, from each individuals' steps"""
    coords = [{"x": [], "y": [], "color": []} for key in range(list_length + 1)]
    for indiv in result:
        for pos_nr, pos in enumerate(result[indiv]["position"]):
            coords[pos_nr]["x"].append(pos[0])
            coords[pos_nr]["y"].append(pos[1])
            coords[pos_nr]["color"].append(dic_color[indiv])
    return coords


## main loop


def initial_population(
    nr_individuals: int,
    nr_of_genes: int,
    nr_of_input: int,
    nr_of_actions: int,
    nr_of_inner: int,
    world_size: int,
):
    """generates list of individuals with genome and brain"""
    individuals = generate_initial_genomes_for_population(
        nr_individuals, nr_of_genes, nr_of_input, nr_of_actions, nr_of_inner
    )

    ## initial brain and position generator
    result = calculate_individual_output_weights(individuals)

    ## add genome
    for indiv in result:
        result[indiv]["genome"] = [i.hex_id for i in individuals[indiv]]

    ## assign position remove brains without output
    pos = generate_random_coords(world_size, nr_individuals)
    assign_position_and_remove_outputless_brains(result, pos)
    return result


def steps_in_generation(
    world_size: int, result: dict, world_size_x: int, world_size_y: int
):
    n = 0
    pbar = tqdm(total=world_size, initial=n)

    while world_size > n:

        pbar.update(1)

        for indiv in result:
            x, y = result[indiv]["position"][-1][0], result[indiv]["position"][-1][1]
            if n < 1:
                calculate_position(result, indiv, x, y, world_size_x, world_size_y)
            elif n >= 1:
                apply_input(result, indiv)
                calculate_position(result, indiv, x, y, world_size_x, world_size_y)

        last_pos_list = {obj: result[obj]["position"][-1] for obj in result}
        prevent_overlap_movement(last_pos_list, result)
        n += 1
    pbar.close()
    return result


def select_individuals_from_safezone(world_size: int, result: dict):
    """select individuals from safe zone and renumerate"""
    safe_zone = int(world_size * 0.6)
    n = 0
    survivors = {}
    for key in result:
        x = result[key]["position"][-1][0]
        if x > safe_zone:
            survivors[n] = {"genome": []}
            survivors[n]["genome"] = result[key]["genome"]
            n += 1
    return survivors, n


def asexual_reproduction_and_mutation(
    world_size: int, result: dict, nr_individuals: int
):
    survivors, n = select_individuals_from_safezone(world_size, result)

    ## reproduct survivors
    new_indiv_len = nr_individuals - list(survivors.keys())[-1]
    new_indiv_nr = np.random.choice(list(survivors.keys()), new_indiv_len - 1)

    for key in new_indiv_nr:
        survivors[n] = {"genome": []}
        survivors[n]["genome"] = survivors[key]["genome"]
        n += 1

    ## mutate population
    for key in survivors:
        genome_mutation(survivors[key]["genome"])

    return survivors


def next_generation(
    survivors: int,
    nr_of_input: int,
    nr_of_actions: int,
    nr_of_inner: int,
    world_size: int,
    nr_individuals: int,
):
    dic = {}
    for nr_idividual in survivors:
        dic[nr_idividual] = gene_to_neuron(
            survivors[nr_idividual]["genome"], nr_of_input, nr_of_actions, nr_of_inner
        )

    result = calculate_individual_output_weights(dic)
    for indiv in result:
        result[indiv]["genome"] = survivors[indiv]["genome"]
    pos = generate_random_coords(world_size, nr_individuals)
    assign_position_and_remove_outputless_brains(result, pos)
    return result
