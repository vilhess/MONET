import sys 
sys.path.append('../../')

import numpy as np
import pandas as pd
import random
from search_algorithms.RandomSearch import RandomSearch, RandomSearch_NTK
from search_spaces.nats_bench_dataset.NATSBenchNode import NATSBenchSizeNode
from nats_bench import create


class RandomSearchNats(RandomSearch):
    def __init__(self, api, df, max_iter):
        super(RandomSearchNats, self).__init__(max_iter)

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
        cell = NATSBenchSizeNode()
        while not cell.is_terminal():
            av_actions = cell.get_action_tuples()
            ac = random.choice(av_actions)
            cell.play_action(ac)
        accuracy, ntk = self.get_reward(cell, self.df)
        return cell, accuracy, ntk

    def get_best_acc(self):
        return np.max(self.df['accuracy'])

class RandomSearchNats_NTK(RandomSearch_NTK, RandomSearchNats):
    def __init__(self, api, df, max_iter):
        RandomSearch_NTK.__init__(self, max_iter)
        RandomSearchNats.__init__(self, api, df, max_iter)
        
    def get_random_cell(self):
        return RandomSearchNats.get_random_cell(self)

    def get_best_ntk_acc(self):
        return np.max(self.df['score']), np.max(self.df['accuracy'])


if __name__=="__main__":
    path = "../Downloads/NATS-sss-v1_0-50262.pickle.pbz2"
    api = create(path, 'sss', fast_mode=False, verbose=False)
    df = pd.read_csv('CSV/nats-c10-mydemo.csv')
    evolution = RandomSearchNats(api, df, 1000)
    evolution.run()
    evolution.plot()

    evolution = RandomSearchNats_NTK(api, df, 1000)
    evolution.run()
    evolution.plot()