import sys
sys.path.append('/Users/samy/Desktop/NTK4NAS/')

import numpy as np
import pandas as pd
import random
from search_spaces.nas_bench_201.NASBench201Node import NASBench201Cell
from nas_201_api import NASBench201API as API
from search_algorithms.LocalSearch import LocalSearch, LocalSearch_NTK


class NB201CellLocalSearch(NASBench201Cell):
    def __init__(self, n_vertices=4):
        super(NB201CellLocalSearch, self).__init__(n_vertices=n_vertices)
        
    
    def get_reward(self, api, df):
        arch_str = self.to_str()
        idx = api.query_index_by_arch(arch_str)
        row = df.loc[df["index"] == idx]
        self.acc = row["accuracy"].item()
        self.ntk = row["score"].item()


class LocalSearchNB201(LocalSearch):
    def __init__(self, api, df):
        super(LocalSearchNB201, self).__init__()

        self.api = api
        self.df = df

    def get_random_cell(self):
        cell = NB201CellLocalSearch()
        while not cell.is_complete():
            av_actions = cell.get_action_tuples()
            ac = random.choice(av_actions)
            cell.play_action(*ac)
        cell.get_reward(self.api, self.df)
        return cell
        
    def get_neighboors(self, model):
        neighboors = model.get_neighboors()
        for neig in neighboors:
            neig.get_reward(self.api, self.df)
        return neighboors

    def get_best_acc(self):
        return np.max(self.df['accuracy'])


class LocalSearchNB201_NTK(LocalSearch_NTK):
    def __init__(self, api, df):
        super(LocalSearchNB201_NTK, self).__init__()

        self.api = api
        self.df = df

    def get_random_cell(self):
        cell = NB201CellLocalSearch()
        while not cell.is_complete():
            av_actions = cell.get_action_tuples()
            ac = random.choice(av_actions)
            cell.play_action(*ac)
        cell.get_reward(self.api, self.df)
        return cell
        
    def get_neighboors(self, model):
        neighboors = model.get_neighboors()
        for neig in neighboors:
            neig.get_reward(self.api, self.df)
        return neighboors



if __name__=="__main__":
    print('loading api')
    api = API("API/NAS-Bench-201-v1_1-096897.pth", verbose=False)
    print('api loaded')
    df = pd.read_csv('benchmark_scores/Cifar10-NB201.csv')

    search = LocalSearchNB201(api, df)
    search.run()
    search.plot()