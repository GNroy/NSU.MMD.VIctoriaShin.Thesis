import numpy as np
from agent import Agent
from itertools import combinations
from joblib import Parallel, delayed
from multiprocessing import cpu_count
from os import fsync


# support function for places generation
def extract(x, y):
    return np.vstack((x//y, x%y))

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

def perform_trial(agents_num, n):
    # prepare agents and shelters positions
    # create agents
    agents = np.array([Agent() for i in range(agents_num)])
    
    #generate places for agents and shelters
    arange = np.random.permutation(n*n).reshape(-1, 1)
    agents_places = extract(arange[:agents_num], n)
    shelters_places = extract(arange[-agents_num:], n)
    #print(agents_places, shelters_places)
    
    # initialize agents with their locations and assigned shelters
    for ap in zip(agents, agents_places, shelters_places):
        ap[0].init_algo(ap[1], agents[agents != ap[0]])
        ap[0].assingn_shelter(ap[-1])
    
    # run the algorithm
    # iterate until everyone finish
    is_finished = np.full(agents_num, 0, dtype=bool)
    round_counter = 0
    while not is_finished.all():
        
        # generate an order of agents calls and run a round
        order_a, order_c = generate_order(agents_num)
        for ord in zip(order_a, order_c):
            agents[ord[0]].round(ord[1])
            #print(agents[ord[0]].NC, agents[ord[0]].CFR)
            if agents[ord[0]].CFR >= 2*(agents_num - 1) and not is_finished[ord[0]]:
                is_finished[ord[0]] = True
        
        round_counter += 1
    
    # perform path correctness check
    if not paths_are_correct(agents):
        raise RuntimeError('Agents paths are incorrect: ' + str(agents_num))
    
    # collect elapsed rounds
    return round_counter

if __name__ == '__main__':

    # expariment variables
    N = 100 # gridsize
    a_min = 2 # minimum number of agents
    a_max = 21 # maximum number of agents
    a_coeff = 1.1 # a way to regulate the agents diversity density
    a_num = np.arange(a_min - 1, a_max - 1, dtype=np.int) + np.logspace(a_min, a_max, a_max - a_min, base=a_coeff).astype(np.int)
    a_num = a_num[a_num <= a_max] # array of agents numbers
    trials_max = 20 # minimum number of trials
    trials_min = 2 # maximum number of trials
    # dictionary of number of trials for each main iteration (depending on agents number)
    # the idea is to use more trials for a small number of agents and less for a large
    trials = dict(zip(a_num, np.logspace(np.log10(trials_min), np.log10(trials_max), len(a_num), dtype=np.int)[::-1]))
    
    # get the number of cores for further use in trials parallelization
    num_cores = cpu_count()
    
    # main cycle (for testing)
    with open('out.log', 'w+') as f:
        for key in trials.keys():
            
            # collect round number for every trial
            # use parallelization
            stat = Parallel(n_jobs=min(num_cores, trials[key]))(delayed(perform_trial)(key, N) for t in range(trials[key]))
            
            # print mean round number for a given agents number
            f.write(' '.join([str(key), str(np.mean(stat))]) + '\n')
            f.flush()
            fsync(f.fileno())
    