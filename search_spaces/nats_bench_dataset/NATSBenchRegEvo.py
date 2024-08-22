import sys 
sys.path.append('../../')

import numpy as np
import pandas as pd
import random
from search_algorithms.RegularizedEvolution import RegularizedEvolution, RegularizedEvolutionNTK
from search_spaces.nats_bench_dataset.NATSBenchNode import NATSBenchSizeNode

from nats_bench import create

class NATSCellRegEvo(NATSBenchSizeNode):
    def __init__(self):
        super(NATSCellRegEvo, self).__init__()

    def get_reward(self, api, df):
        arch_str = self.to_str()
        idx = api.query_index_by_arch(arch_str)
        row = df.loc[df["index"] == idx]
        self.acc = row["accuracy"].item()
        self.ntk = np.log(row["score"].item())
        

class RegularizedEvolutionNATS(RegularizedEvolution):
    def __init__(self, api, df, population_size=500, cycles=1000, sample_size=25):
        super(RegularizedEvolutionNATS, self).__init__(population_size, cycles, sample_size)

        self.api = api
        self.df = df
        
    def get_random_cell(self):
        cell = NATSCellRegEvo()
        while not cell.is_terminal():
            av_actions = cell.get_action_tuples()
            ac = random.choice(av_actions)
            cell.play_action(ac)
        cell.get_reward(self.api, self.df)
        return cell

    def mutate(self, model):
        mutated = model.mutate()
        mutated.get_reward(self.api, self.df)
        return mutated

    def get_best_acc(self):
        return np.max(self.df['accuracy'])

class RegularizedEvolutionNATS_NTK(RegularizedEvolutionNTK):
    def __init__(self, api, df, population_size=500, cycles=1000, sample_size=25):
        super(RegularizedEvolutionNATS_NTK, self).__init__(population_size, cycles, sample_size)

        self.api = api
        self.df = df
        
    def get_random_cell(self):
        cell = NATSCellRegEvo()
        while not cell.is_terminal():
            av_actions = cell.get_action_tuples()
            ac = random.choice(av_actions)
            cell.play_action(ac)
        cell.get_reward(self.api, self.df)
        return cell

    def mutate(self, model):
        mutated = model.mutate()
        mutated.get_reward(self.api, self.df)
        return mutated

    def get_best_ntk_acc(self):
        return np.log(np.max(self.df['score'])), np.max(self.df['accuracy'])


if __name__=="__main__":
    path = "../Downloads/NATS-sss-v1_0-50262.pickle.pbz2"
    api = create(path, 'sss', fast_mode=False, verbose=False)
    df = pd.read_csv('CSV/nats-c10-mydemo.csv')
    evolution = RegularizedEvolutionNATS(api, df)
    evolution.run()
    evolution.plot()
