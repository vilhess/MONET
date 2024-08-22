import sys
sys.path.append('/Users/samy/Desktop/NTK4NAS/')

import numpy as np
import random
from copy import deepcopy
from nasbench import api

from search_algorithms.RegularizedEvolution import RegularizedEvolution

INPUT = 'input'
OUTPUT = 'output'
CONV3X3 = 'conv3x3-bn-relu'
CONV1X1 = 'conv1x1-bn-relu'
MAXPOOL3X3 = 'maxpool3x3'
NUM_VERTICES = 7
MAX_EDGES = 9
EDGE_SPOTS = NUM_VERTICES * (NUM_VERTICES - 1) / 2   # Upper triangular matrix
OP_SPOTS = NUM_VERTICES - 2   # Input/output vertices are fixed
ALLOWED_OPS = [CONV3X3, CONV1X1, MAXPOOL3X3]
ALLOWED_EDGES = [0, 1]   # Binary adjacency matrix

class NB101Model:
    def __init__(self, spec, api):
        self.spec = spec
        self.api = api

    def mutate(self, mutation_rate=1):
        old_spec = self.spec
        while True:
            new_matrix = deepcopy(old_spec.original_matrix)
            new_ops = deepcopy(old_spec.original_ops)

            edge_mutation_prob = mutation_rate/NUM_VERTICES
            for src in range(0, NUM_VERTICES-1):
                for dst in range(src+1, NUM_VERTICES):
                    if random.random()<edge_mutation_prob:
                        new_matrix[src, dst] = 1 - new_matrix[src, dst]
            
            op_mutation_prob = mutation_rate/OP_SPOTS
            for ind in range(1, NUM_VERTICES-1):
                if random.random()<op_mutation_prob:
                    available=[o for o in self.api.config['available_ops'] if o != new_ops[ind]]
                    new_ops[ind]=random.choice(available)
            new_spec = api.ModelSpec(new_matrix, new_ops)
            if self.api.is_valid(new_spec):
                mutated = new_spec
                return new_spec

    def get_reward(self, nasbench):
        info = nasbench.query(self.spec)
        acc = info['validation_accuracy']
        self.acc = acc

    # def get_best_acc
        


class RegularizedEvolutionNB101(RegularizedEvolution):
    def __init__(self, api, population_size=3000, cycles=1000, sample_size=2000):
        super(RegularizedEvolutionNB101, self).__init__(population_size, cycles, sample_size)
        self.api = api
        
    def get_random_cell(self):
        """Returns a random valid spec."""
        while True:
            matrix = np.random.choice(ALLOWED_EDGES, size=(NUM_VERTICES, NUM_VERTICES))
            matrix = np.triu(matrix, 1)
            ops = np.random.choice(ALLOWED_OPS, size=(NUM_VERTICES)).tolist()
            ops[0] = INPUT
            ops[-1] = OUTPUT
            spec = api.ModelSpec(matrix=matrix, ops=ops)
            if self.api.is_valid(spec):
                model = NB101Model(spec, self.api)
                model.get_reward(self.api)
                return model

    def mutate(self, cell, mutation_rate=1):
        mutated = cell.mutate()
        mutated = NB101Model(mutated, self.api)
        mutated.get_reward(self.api)
        return mutated

if __name__=="__main__":

    nasbench = api.NASBench("API/nasbench_full.tfrecord")

    INPUT = 'input'
    OUTPUT = 'output'
    CONV3X3 = 'conv3x3-bn-relu'
    CONV1X1 = 'conv1x1-bn-relu'
    MAXPOOL3X3 = 'maxpool3x3'
    NUM_VERTICES = 7
    MAX_EDGES = 9
    EDGE_SPOTS = NUM_VERTICES * (NUM_VERTICES - 1) / 2   # Upper triangular matrix
    OP_SPOTS = NUM_VERTICES - 2   # Input/output vertices are fixed
    ALLOWED_OPS = [CONV3X3, CONV1X1, MAXPOOL3X3]
    ALLOWED_EDGES = [0, 1]   # Binary adjacency matrix

    evolution = RegularizedEvolutionNB101(nasbench)
    evolution.run()
    evolution.plot()