from genetic_algorithm import *
from numpy.random import default_rng

# ## hexadecimal generator

nr_of_input = 3
nr_of_actions = 3
nr_of_inner = 3
nr_of_genes = 16
nr_individuals = 4

individuals = generate_initial_genomes_for_population(nr_individuals, nr_of_genes, nr_of_input, nr_of_actions, nr_of_inner)

## world size
world_size_x = 128
world_size_y = 128

rng = default_rng()
x = rng.choice(world_size_x, size=nr_individuals, replace=False)
y = rng.choice(world_size_y, size=nr_individuals, replace=False)

# ## initial brain and position generator

result = calculate_individual_output_weights(individuals)
print(result)

a = [(3, 0),(4,1),(5,2)]

def slope_intercept(x1,y1,x2,y2):
    a = (y2 - y1) / (x2 - x1)
    b = y1 - a * x1     
    return a,b

# print(slope_intercept(3,0,2,1))

if a[-2][0] == a[-1][0]:
    x = a[-1][0]
elif a[-2][0] > a[-1][0]:
    x = a[-2][0] + 1
elif a[-2][0] < a[-1][0]:
    x = a[-2][0] - 1
# print(x)

a[-2][0]

