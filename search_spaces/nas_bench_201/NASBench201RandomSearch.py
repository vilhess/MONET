import sys
sys.path.append('/Users/samy/Desktop/NTK4NAS/')

import numpy as np
import pandas as pd
import random
from search_algorithms.RandomSearch import RandomSearch, RandomSearch_NTK
from search_spaces.nas_bench_201.NASBench201Node import NASBench201Cell
from nas_201_api import NASBench201API as API


class NB201CellRandomSearch(NASBench201Cell):
    def __init__(self, n_vertices=4):
        super(NB201CellRandomSearch, self).__init__(n_vertices=n_vertices)
        
    
    def get_reward(self, api, df):
        arch_str = self.to_str()
        idx = api.query_index_by_arch(arch_str)
        row = df.loc[df["index"] == idx]
        self.acc = row["accuracy"].item()
        self.ntk = row["score"].item()
        

class RandomSearchNB201(RandomSearch):
    def __init__(self, api, df, max_iter):
        super(RandomSearchNB201, self).__init__(max_iter)
        self.api = api
        self.df = df

    def get_reward(self, cell, df):
        arch_str = cell.to_str()
        idx = self.api.query_index_by_arch(arch_str)
        row = df.loc[df["index"] == idx]
        acc = row["accuracy"].item()
        ntk = row["score"].item()
        return acc, ntk
        
    def get_random_cell(self):
        cell = NASBench201Cell()
        while not cell.is_complete():
            av_actions = cell.get_action_tuples()
            ac = random.choice(av_actions)
            cell.play_action(*ac)
        accuracy, ntk = self.get_reward(cell, self.df)
        return cell, accuracy, ntk

    def get_best_acc(self):
        return np.max(self.df['accuracy'])

class RandomSearchNB201_NTK(RandomSearch_NTK, RandomSearchNB201):
    def __init__(self, api, df, max_iter):
        RandomSearch_NTK.__init__(self, max_iter)
        RandomSearchNB201.__init__(self, api, df, max_iter)
        
    def get_random_cell(self):
        return RandomSearchNB201.get_random_cell(self)

    def run(self):
        return RandomSearch_NTK.run(self)

    def get_best_ntk_acc(self):
        return np.max(self.df['score']), np.max(self.df['accuracy'])


if __name__=="__main__":
    print("Loading API")
    api = API("API/NAS-Bench-201-v1_1-096897.pth", verbose=False)
    print("API Loaded")
    df = pd.read_csv('benchmark_scores/Cifar10-NB201.csv')
    evolution = RandomSearchNB201(api, df, 1000)
    evolution.run()
    evolution.plot()

    evolution = RandomSearchNB201_NTK(api, df, 1000)
    evolution.run()
    evolution.plot()



