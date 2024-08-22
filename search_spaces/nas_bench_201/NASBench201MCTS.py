import sys
sys.path.append('/Users/samy/Desktop/NTK4NAS/')

import copy
import random

import time
import json

import numpy as np
from torch import nn
from xautodl.models import get_cell_based_tiny_net

from MCTS.mcts_agent import UCT, NTKScorer
from MCTS.Node import Node
from MCTS.nested import NestedMCS, NRPA, NestedMCS_NTK, NRPA_NTK
from search_spaces.nas_bench_201.NASBench201Node import NASBench201Node, NASBench201Cell, NASBench201AMAFNode, NASBench201NestedNode

N_NODES = 4
ADJACENCY_MATRIX_SIZE = N_NODES ** 2
N_OPERATIONS = 5


class NASBench201UCT(UCT):

    def __init__(self, root_node: NASBench201Node, api, save_folder=None, params_path=None, disable_tqdm=False, df=None):
        super().__init__(root_node, save_folder=save_folder, params_path=params_path, disable_tqdm=disable_tqdm)
        self.zobrist_table = []
        for i in range(ADJACENCY_MATRIX_SIZE):
            adjacency_table = []
            for operation in range(N_OPERATIONS):
                adjacency_table.append(random.randint(0, 2 ** 64))
            self.zobrist_table.append(adjacency_table)
        self.hash_table = {}
        self.root.add_zobrist(self.zobrist_table)
        self.api = api
        self.df = df
        self.metric = "accuracy"
        self.accuracies_tracker = []
        self.best_accuracy_val = 0

    def _get_reward(self, node: Node):
        if self.df is not None:
            arch_str = node.state.to_str()
            idx = self.api.query_index_by_arch(arch_str)
            row = self.df.loc[self.df["index"] == idx]
            reward = row[self.metric].item()
            ntk = row["score"].item()
            if reward > self.best_reward_value:
                self.best_accuracy_val = row["accuracy"].item()
            self.accuracies_tracker.append(self.best_accuracy_val)
            return reward
        # 1. Find the string associated to the current state
        state_str = node.state.to_str()
        # 2. Find the associated architecture index in the api
        index = self.api.query_index_by_arch(state_str)
        # 3. Fetch desired metric from API.
        info = self.api.query_meta_info_by_index(index, hp="200")
        reward = info.get_metrics("cifar10-valid", "valid")["accuracy"] / 100
        #reward = self.api.get_more_info(index, 'cifar100', None, 200, False)["valid-accuracy"] / 100
        # print(f"[PLAYOUT] reward = {reward}")
        return reward


class NASBench201UCT_NTK(NASBench201UCT, NTKScorer):

    def __init__(self, root_node: NASBench201Node, api, save_folder=None, params_path=None, disable_tqdm=False, df=None):
        super(NASBench201UCT_NTK, self).__init__(root_node, api, save_folder=save_folder, params_path=params_path,
                                                 disable_tqdm=disable_tqdm, df=df)

    def read_params(self, path):
        NASBench201UCT.read_params(self, path)
        NTKScorer.read_params(self, path)

    def _get_reward(self, node: Node):
        network = self._create_network(node)
        reward = self._score_network(network)
        return reward

    def _playout(self, node: Node):
        """
        Crée un playout aléatoire et renvoie l'accuracy sur le modèle entraîné
        :return:
        """
        node_type = type(node)
        playout_node = node_type(state=copy.deepcopy(node.state))

        while not playout_node.is_terminal():
            available_actions = playout_node.get_action_tuples()
            random_action = available_actions[np.random.randint(len(available_actions))]
            playout_node.play_action(random_action)
            # print(f"[PLAYOUT] Playing random action {random_action}")

        reward = self._get_reward(playout_node)
        accuracy = NASBench201UCT._get_reward(self, playout_node)

        del playout_node
        return reward, accuracy

    def next_best_move(self, all_rewards=None, best_reward=None, all_accuracies=None, best_accuracy=None) -> Node:
        return NTKScorer.next_best_move(self, all_rewards, best_reward, all_accuracies, best_accuracy)

    def main_loop(self):
        return NTKScorer.main_loop(self)

    def _create_network(self, node: Node) -> nn.Module:
        # unet = NASBench201UNet(node.state.to_str(), input_size=128, input_depth=1)
        # network = NASBench201UNet_NTK(node.state.to_str(), input_size=128, input_depth=1)
        index = self.api.query_index_by_arch(node.state.to_str())
        config = self.api.get_net_config(index,
                                         'cifar10')  # obtain the network configuration for the 123-th architecture on the CIFAR-10 dataset
        network = get_cell_based_tiny_net(config)  # create the network from configuration
        return network

    def _score_network(self, network):
        return NTKScorer._score_network(self, network, benchmark="nb201")


class NASBench201NestedMCS(NestedMCS, NASBench201UCT):

    def __init__(self, root_node: NASBench201NestedNode, level, api, save_folder=None, params_path=None,
                 disable_tqdm=False, df=None):
        NestedMCS.__init__(self, root_node, level, save_folder=save_folder,
                           disable_tqdm=disable_tqdm, params_path=params_path)
        NASBench201UCT.__init__(self, root_node, api, save_folder, params_path, disable_tqdm, df=df)

    def main_loop(self):
        return NestedMCS.main_loop(self)

    def _get_reward(self, node: Node):
        return NASBench201UCT._get_reward(self, node)


class NASBench201NRPA(NRPA, NASBench201UCT):

    def __init__(self, root_node: NASBench201NestedNode, level, api, save_folder=None, params_path=None,
                 disable_tqdm=False, df=None):
        NRPA.__init__(self, root_node, level, save_folder=save_folder,
                      disable_tqdm=disable_tqdm, params_path=params_path)
        NASBench201UCT.__init__(self, root_node, api, save_folder, params_path, disable_tqdm, df)

    def _get_reward(self, node: Node):
        return NASBench201UCT._get_reward(self, node)


class NASBench201NestedMCS_NTK(NASBench201NestedMCS, NASBench201UCT_NTK, NestedMCS_NTK):

    def __init__(self, root_node: NASBench201NestedNode, level, api, save_folder=None, params_path=None,
                 disable_tqdm=False, df=None):
        NestedMCS_NTK.__init__(self, root_node, level, save_folder, disable_tqdm, params_path)
        NASBench201NestedMCS.__init__(self, root_node, level, api, save_folder, params_path, disable_tqdm, df)
        self.class_for_accuracy = NASBench201NestedMCS

    def _get_reward(self, node: Node):
        return NASBench201UCT_NTK._get_reward(self, node)

    def _playout(self, node: Node):
        return NestedMCS_NTK._playout(self, node)

    def main_loop(self):
        return NestedMCS_NTK.main_loop(self)


class NASBench201NRPA_NTK(NASBench201NRPA, NASBench201UCT_NTK, NRPA_NTK):

    def __init__(self, root_node: NASBench201NestedNode, level, api, save_folder=None, params_path=None,
                 disable_tqdm=False, df=None):
        NRPA_NTK.__init__(self, root_node, level, save_folder, disable_tqdm, params_path)
        NASBench201NRPA.__init__(self, root_node, level, api, save_folder=save_folder,
                                 disable_tqdm=disable_tqdm, params_path=params_path, df=df)
        self.class_for_accuracy = NASBench201NRPA

    def read_params(self, path):
        NASBench201NRPA.read_params(self, path)
        NASBench201UCT_NTK.read_params(self, path)
        NRPA_NTK.read_params(self, path)

    def _playout(self, node: NASBench201NestedNode):
        return NRPA_NTK._playout(self, node)

    def _get_reward(self, node: Node):
        return NASBench201UCT_NTK._get_reward(self, node)

    def main_loop(self):
        return NRPA_NTK.main_loop(self)


if __name__ == '__main__':
    node = NASBench201AMAFNode(NASBench201Cell(4))
