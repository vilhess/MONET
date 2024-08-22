import numpy as np
from nasbench import api

from search_algorithms.RandomSearch import RandomSearch

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
    def __init__(self, spec):
        self.spec = spec

    def get_reward(self, nasbench):
        info = nasbench.query(self.spec)
        acc = info['validation_accuracy']
        self.acc = acc
        return acc
        


class RandomSearchNASBench101(RandomSearch):
    def __init__(self, api, max_iter=1000):
        super(RandomSearchNASBench101, self).__init__(max_iter)
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
                model = NB101Model(spec)
                acc = model.get_reward(self.api)
                return model, acc, None

if __name__=="__main__":

    nasbench = api.NASBench("../Downloads/nasbench_full.tfrecord")

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

    evolution = RandomSearchNASBench101(nasbench)
    evolution.run()
    evolution.plot()