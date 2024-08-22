import sys 
sys.path.append('../../')

import numpy as np
import pandas as pd
import random
from nats_bench_dataset.NATSBenchNode import NATSBenchSizeNode
from nats_bench import create
from search_algorithms.LocalSearch import LocalSearch, LocalSearch_NTK

class NATSCellLocalSearch(NATSBenchSizeNode):
    def __init__(self):
        super(NATSCellLocalSearch, self).__init__()

    def get_reward(self, api, df):
        arch_str = self.to_str()
        idx = api.query_index_by_arch(arch_str)
        row = df.loc[df["index"] == idx]
        self.acc = row["accuracy"].item()
        self.ntk = np.log(row["score"].item())

class LocalSearchNATS(LocalSearch):
    def __init__(self, api, df):
        super(LocalSearchNATS, self).__init__()

        self.api = api
        self.df = df

    def get_random_cell(self):
        cell = NATSCellLocalSearch()
        while not cell.is_terminal():
            av_actions = cell.get_action_tuples()
            ac = random.choice(av_actions)
            cell.play_action(ac)
        cell.get_reward(self.api, self.df)
        return cell
        
    def get_neighboors(self, model):
        neighboors = model.get_neighboors()
        for neig in neighboors:
            neig.get_reward(self.api, self.df)
        return neighboors

    def get_best_acc(self):
        return np.max(self.df['accuracy'])


class LocalSearchNATS_NTK(LocalSearch_NTK):
    def __init__(self, api, df):
        super(LocalSearchNATS_NTK, self).__init__()

        self.api = api
        self.df = df

    def get_random_cell(self):
        cell = NATSCellLocalSearch()
        while not cell.is_terminal():
            av_actions = cell.get_action_tuples()
            ac = random.choice(av_actions)
            cell.play_action(ac)
        cell.get_reward(self.api, self.df)
        return cell
        
    def get_neighboors(self, model):
        neighboors = model.get_neighboors()
        for neig in neighboors:
            neig.get_reward(self.api, self.df)
        return neighboors

    def get_best_acc(self):
        return np.max(self.df['accuracy'])

if __name__=="__main__":
    path = "../Downloads/NATS-sss-v1_0-50262.pickle.pbz2"
    api = create(path, 'sss', fast_mode=False, verbose=False)
    df = pd.read_csv('CSV/nats-c10-mydemo.csv')
    search = LocalSearchNATS(api, df)
    search.run()
    search.plot()