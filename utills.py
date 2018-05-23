import numpy as np
from agent import Agent
from itertools import combinations


# support function for places generation
def extract(x, y):
    return np.hstack((x//y, x%y))

# support function for order generation
def generate_order(m):
    arr = np.arange(m)
    order_a = np.zeros(m, dtype=np.int)
    order_c = np.zeros(m, dtype=object)
    for i in range(m):
        order_a[i] = np.random.choice(arr)
        arr = arr[arr != order_a[i]]
        order_c[i] = np.random.permutation(arr)
        order_c[i][order_c[i] > order_a[i]] -= 1
    return order_a, order_c

# support function for supervised path correctness check
def paths_are_correct(agents):
    for item in combinations(agents, 2):
        s11 = item[0].distance
        s12 = item[0].get_distance(item[1].shelter)
        s21 = item[1].get_distance(item[0].shelter)
        s22 = item[1].distance
        if s11 + s22 > s12 + s21:
            return False
    return True

# function for agents initialization with shelters assigned
def model_init(agents_num, n):
    # create agents
    agents = np.array([Agent() for i in range(agents_num)])
    
    #generate places for agents and shelters
    arange = np.random.permutation(n*n).reshape(-1, 1)
    agents_places = extract(arange[:agents_num], n)
    shelters_places = extract(arange[-agents_num:], n)
    
    # initialize agents with their locations and assigned shelters
    for ap in zip(agents, agents_places, shelters_places):
        ap[0].init_algo(ap[1], agents[agents != ap[0]])
        ap[0].assingn_shelter(ap[-1])
    
    return agents