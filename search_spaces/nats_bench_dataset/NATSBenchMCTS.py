import random
import copy
import numpy as np

from torch import nn
from xautodl.models import get_cell_based_tiny_net

from MCTS import Node
from MCTS.mcts_agent import UCT, RAVE, GRAVE, NTKScorer
from MCTS.nested import NestedMCS, NRPA, NestedMCS_NTK, NRPA_NTK
from search_spaces.nats_bench_dataset.NATSBenchNode import NATSBenchSizeNode, NATSBenchSizeAMAFNode, NATSBenchSizeNestedNode

ADJACENCY_MATRIX_SIZE = 5
N_OPERATIONS = 8


class NATSBenchUCT(UCT):

    def __init__(self, root_node: NATSBenchSizeNode, api, save_folder=None, params_path=None, disable_tqdm=False, df=None):
        UCT.__init__(self, root_node, save_folder, disable_tqdm, params_path=params_path)
        self.api = api
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
        # self.accuracies_tracker = []
        # self.best_accuracy = 0

    def _get_reward(self, node: NATSBenchSizeNode):
        if self.df is not None:
            arch_str = node.to_str()
            idx = self.api.query_index_by_arch(arch_str)
            row = self.df.loc[self.df["index"] == idx]
            reward = row[self.metric].item()
            ntk = row["score"].item()
            # if reward > self.best_reward_value:
            #     self.best_accuracy = row["accuracy"].item()
            # self.accuracies_tracker.append(self.best_accuracy)
            return reward
        # 1. Find the string associated to the current state
        state_str = node.to_str()
        # 2. Find the associated architecture index in the api
        index = self.api.query_index_by_arch(state_str)
        # 3. Fetch desired metric from API.
        reward = self.api.get_more_info(index, 'cifar100', hp="90")["test-accuracy"]
        # print(f"[PLAYOUT] reward = {reward}")
        return reward


class NATSBenchRAVE(NATSBenchUCT, RAVE):

    def __init__(self, root_node: NATSBenchSizeAMAFNode, api, save_folder=None, disable_tqdm=False, params_path=None):
        NATSBenchUCT.__init__(self, root_node=root_node,
                              api=api,
                              save_folder=save_folder,
                              params_path=params_path,
                              disable_tqdm=disable_tqdm)
        RAVE.__init__(self, root_node, params_path, save_folder, disable_tqdm)

    def _selection(self, node: Node) -> Node:
        """
            Selects a candidate child node from the input node.
            """
        return RAVE._selection(self, node)

    def _expansion(self, node: Node) -> Node:
        """
            Unless L ends the game decisively (e.g. win/loss/draw) for either player,
            create one (or more) child nodes and choose node C from one of them.
            Child nodes are any valid moves from the game position defined by L.
            """
        return RAVE._expansion(self, node)

    def _playout(self, node: Node):
        """
            Crée un playout aléatoire et renvoie l'accuracy sur le modèle entraîné
            :return:
            """

        return NATSBenchUCT._playout(self, node)

    def _backpropagation(self, node: Node, result: float):
        """
            Backpropagates the result of a playout up the tree.
            """
        return RAVE._backpropagation(self, node, result)

    def main_loop(self) -> Node:
        """
            Body of UCT
            """
        return NATSBenchUCT.main_loop(self)


class NATSBenchGRAVE(NATSBenchRAVE, GRAVE):

    def __init__(self, root_node: Node, api, save_folder=None, params_path=None, disable_tqdm=False):
        super().__init__(root_node, api, save_folder=save_folder, params_path=params_path, disable_tqdm=disable_tqdm)

    def _selection(self, node: Node) -> Node:
        """
        Selects a candidate child node from the input node.
        """
        return GRAVE._selection(self, node)


class NATSBenchNestedMCS(NestedMCS, NATSBenchUCT):

    def __init__(self, root_node: NATSBenchSizeNestedNode, level, api, save_folder=None, params_path=None,
                 disable_tqdm=False):
        NestedMCS.__init__(self, root_node, level, save_folder=save_folder,
                           disable_tqdm=disable_tqdm, params_path=params_path)
        NATSBenchUCT.__init__(self, root_node, api, save_folder, params_path, disable_tqdm)

    def main_loop(self):
        return NestedMCS.main_loop(self)

    def _get_reward(self, node: Node):
        return NATSBenchUCT._get_reward(self, node)


class NATSBenchNRPA(NRPA, NATSBenchUCT):

    def __init__(self, root_node: NATSBenchSizeNestedNode, level, api, save_folder=None, params_path=None,
                 disable_tqdm=False, df=None):
        NRPA.__init__(self, root_node, level, save_folder=save_folder,
                      disable_tqdm=disable_tqdm, params_path=params_path)
        NATSBenchUCT.__init__(self, root_node, api, save_folder, params_path, disable_tqdm, df)

    def _code(self, node, move):
        if move == None:
            ### SEULEMENT POUR LA RACINE DE L'ARBRE A PRIORI
            return node.hash

        state_code = node.hash
        state_code = ""
        code = str(state_code) + str(move[0]) + str(move[1])

        return code

    def _score_node(self, child: Node, parent: Node, C=None) -> float:
        return NATSBenchUCT._score_node(self, child, parent, C)


class NATSBenchUCT_NTK(NATSBenchUCT, NTKScorer):

    def __init__(self, root_node: NATSBenchSizeNode, api, save_folder=None, params_path=None, disable_tqdm=False):
        super(NATSBenchUCT_NTK, self).__init__(root_node, api, save_folder=save_folder, params_path=params_path,
                                               disable_tqdm=disable_tqdm)

    def read_params(self, path):
        NATSBenchUCT.read_params(self, path)
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
        accuracy = NATSBenchUCT._get_reward(self, playout_node)

        del playout_node
        return reward, accuracy

    def next_best_move(self, all_rewards=None, best_reward=None, all_accuracies=None, best_accuracy=None) -> Node:
        return NTKScorer.next_best_move(self, all_rewards, best_reward, all_accuracies, best_accuracy)

    def main_loop(self):
        return NTKScorer.main_loop(self)

    def _create_network(self, node: Node) -> nn.Module:
        # unet = NATSBenchUNet(node.state.to_str(), input_size=128, input_depth=1)
        # network = NATSBenchUNet_NTK(node.state.to_str(), input_size=128, input_depth=1)
        index = self.api.query_index_by_arch(node.to_str())
        config = self.api.get_net_config(index,
                                         'cifar10')  # obtain the network configuration for the 123-th architecture on the CIFAR-10 dataset
        network = get_cell_based_tiny_net(config)  # create the network from configuration
        return network

    def _score_network(self, network):
        return NTKScorer._score_network(self, network, benchmark="nats")


class NATSBenchRAVE_NTK(NATSBenchUCT_NTK, NATSBenchRAVE):

    def __init__(self, root_node: NATSBenchSizeAMAFNode, api, save_folder=None, params_path=None, disable_tqdm=False):
        super(NATSBenchRAVE_NTK, self).__init__(root_node, api, save_folder=save_folder, params_path=params_path,
                                                disable_tqdm=disable_tqdm)

    def read_params(self, path):
        NATSBenchUCT_NTK.read_params(self, path)
        NATSBenchRAVE.read_params(self, path)

    def _selection(self, node: Node) -> Node:
        return NATSBenchRAVE._selection(self, node)

    def _expansion(self, node: Node) -> Node:
        return NATSBenchRAVE._expansion(self, node)

    def _playout(self, node: Node):
        return NATSBenchUCT_NTK._playout(self, node)

    def _backpropagation(self, node: Node, result: float):
        return NATSBenchRAVE._backpropagation(self, node, result)


class NATSBenchGRAVE_NTK(NATSBenchRAVE_NTK, NATSBenchGRAVE):

    def __init__(self, root_node: NATSBenchSizeAMAFNode, api, save_folder=None, params_path=None, disable_tqdm=False):
        super(NATSBenchGRAVE_NTK, self).__init__(root_node, api, save_folder=save_folder, params_path=params_path,
                                                 disable_tqdm=disable_tqdm)

    def read_params(self, path):
        NATSBenchRAVE_NTK.read_params(self, path)
        NATSBenchGRAVE.read_params(self, path)

    def _selection(self, node: Node) -> Node:
        return NATSBenchGRAVE._selection(self, node)

    def _expansion(self, node: Node) -> Node:
        return NATSBenchGRAVE._expansion(self, node)

    def _playout(self, node: Node):
        return NATSBenchRAVE_NTK._playout(self, node)

    def _backpropagation(self, node: Node, result: float):
        return NATSBenchGRAVE._backpropagation(self, node, result)


class NATSBenchNestedMCS_NTK(NATSBenchNestedMCS, NATSBenchUCT_NTK, NestedMCS_NTK):

    def __init__(self, root_node: NATSBenchSizeNestedNode, level, api, save_folder=None, params_path=None,
                 disable_tqdm=False):
        NestedMCS_NTK.__init__(self, root_node, level, save_folder, disable_tqdm, params_path)
        NATSBenchNestedMCS.__init__(self, root_node, level, api, save_folder, params_path, disable_tqdm)
        self.class_for_accuracy = NATSBenchNestedMCS

    def _get_reward(self, node: Node):
        return NATSBenchUCT_NTK._get_reward(self, node)

    def _playout(self, node: Node):
        return NestedMCS_NTK._playout(self, node)

    def main_loop(self):
        return NestedMCS_NTK.main_loop(self)


class NATSBenchNRPA_NTK(NATSBenchNRPA, NATSBenchUCT_NTK, NRPA_NTK):

    def __init__(self, root_node: NATSBenchSizeNestedNode, level, api, save_folder=None, params_path=None,
                 disable_tqdm=False):
        NRPA_NTK.__init__(self, root_node, level, save_folder, disable_tqdm, params_path)
        NATSBenchNRPA.__init__(self, root_node, level, api, save_folder=save_folder,
                               disable_tqdm=disable_tqdm, params_path=params_path)
        self.class_for_accuracy = NATSBenchNRPA

    def read_params(self, path):
        NATSBenchNRPA.read_params(self, path)
        NATSBenchUCT_NTK.read_params(self, path)
        NRPA_NTK.read_params(self, path)

    def _playout(self, node: NATSBenchSizeNestedNode):
        return NRPA_NTK._playout(self, node)

    def _get_reward(self, node: Node):
        return NATSBenchUCT_NTK._get_reward(self, node)

    def main_loop(self):
        return NRPA_NTK.main_loop(self)
