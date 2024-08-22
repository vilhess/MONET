import sys
sys.path.append('../../')

import numpy as np
import pandas as pd
import random
from search_algorithms.RegularizedEvolution import RegularizedEvolution, RegularizedEvolutionNTK
from search_spaces.nas_bench_201.NASBench201Node import NASBench201Cell
from nas_201_api import NASBench201API as API

class NB201CellRegEvo(NASBench201Cell):
    def __init__(self, n_vertices=4):
        super(NB201CellRegEvo, self).__init__(n_vertices=n_vertices)
        
    
    def get_reward(self, api, df):
        arch_str = self.to_str()
        idx = api.query_index_by_arch(arch_str)
        row = df.loc[df["index"] == idx]
        self.acc = row["accuracy"].item()
        self.ntk = row["score"].item()
        

class RegularizedEvolutionNB201(RegularizedEvolution):
    def __init__(self, api, df, population_size=5000, cycles=10000, sample_size=5000):
        super(RegularizedEvolutionNB201, self).__init__(population_size, cycles, sample_size)

        self.api = api
        self.df = df
        
    def get_random_cell(self):
        cell = NB201CellRegEvo()
        while not cell.is_complete():
            av_actions = cell.get_action_tuples()
            ac = random.choice(av_actions)
            cell.play_action(*ac)
        cell.get_reward(self.api, self.df)
        return cell

    def mutate(self, model):
        mutated = model.mutate()
        mutated.get_reward(self.api, self.df)
        return mutated

    def get_best_acc(self):
        return np.max(self.df['accuracy'])


class RegularizedEvolutionNB201_NTK(RegularizedEvolutionNTK):
    def __init__(self, api, df, population_size=50, cycles=10000, sample_size=5000):
        super(RegularizedEvolutionNB201_NTK, self).__init__(population_size, cycles, sample_size)

        self.api = api
        self.df = df
        
    def get_random_cell(self):
        cell = NB201CellRegEvo()
        while not cell.is_complete():
            av_actions = cell.get_action_tuples()
            ac = random.choice(av_actions)
            cell.play_action(*ac)
        cell.get_reward(self.api, self.df)
        return cell

    def mutate(self, model):
        mutated = model.mutate()
        mutated.get_reward(self.api, self.df)
        return mutated

    def get_best_ntk_acc(self):
        return np.log(np.max(self.df['score'])), np.max(self.df['accuracy'])



if __name__=="__main__":
    print('loading api')
    api = API('/userdata/T0259728/Bureau/NAS-Bench-201-v1_1-096897.pth', verbose=False)
    print('api loaded')
    df = pd.read_csv('../CSV/Cifar10-my-proof.csv')
    evolution = RegularizedEvolutionNB201(api, df)
    evolution.run()
    evolution.plot()
