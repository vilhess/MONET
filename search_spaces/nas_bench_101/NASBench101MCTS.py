import copy
import random

import nasbench
from nasbench import api as ModelSpecAPI
from nasbench_pytorch.model import Network

import numpy as np
from scipy.special import softmax
from torch import nn

from MCTS.Node import Node
from MCTS.mcts_agent import UCT, RAVE, GRAVE, NTKScorer
from MCTS.nested import NestedMCS, NRPA, NRPA_NTK
from search_spaces.nas_bench_101 import NASBench101Node
from search_spaces.nas_bench_101.NASBench101Node import NASBench101AMAFNode, NASBench101NestedNode, NASBench101Cell

N_NODES = 7
ADJACENCY_MATRIX_SIZE = N_NODES ** 2
N_OPERATIONS = 6

class NASBench101UCT(UCT):

    def __init__(self, root_node: NASBench101Node, api: nasbench.api, save_folder=None, params_path=None,
                 disable_tqdm=False):
        super().__init__(root_node, save_folder=save_folder, disable_tqdm=disable_tqdm, params_path=params_path)
        # self.df = pd.read_csv("/userdata/T0259728/projets/nas_ntk/nas_bench_101/nas_bench_101.csv")
        self.api = api
        self.zobrist_table = []
        for i in range(ADJACENCY_MATRIX_SIZE):
            adjacency_table = []
            for connexion in range(2):  # Connextion or no connexion
                adjacency_table.append(random.randint(0, 2 ** 64))
            self.zobrist_table.append(adjacency_table)
        for i in range(N_NODES):
            operation_table = []
            for operation in range(N_OPERATIONS):
                operation_table.append(random.randint(0, 2 ** 64))
            self.zobrist_table.append(operation_table)
        self.hash_table = {}
        self.root.add_zobrist(self.zobrist_table)

    def _get_reward(self, node: NASBench101Node):
        adjacency, operations = node.state.operations_and_adjacency()
        model_spec = ModelSpecAPI.ModelSpec(
                # Adjacency matrix of the module
                matrix=adjacency,
                # Operations at the vertices of the module, matches order of matrix
                ops=operations)
        # fixed, computed = self.api.get_metrics_from_spec(model_spec)
        # test_acc = np.mean([e["final_test_accuracy"] for e in computed[108]])
        info = self.api.query(model_spec)
        test_acc = info['validation_accuracy']
        return test_acc

    def _playout(self, node: NASBench101Node):

        """
        Crée un playout aléatoire et renvoie l'accuracy sur le modèle entraîné
        :return:
        """
        is_valid = False
        i = 0
        while not is_valid:
            i += 1
            node_type = type(node)
            playout_node = node_type(state=copy.deepcopy(node.state))
            while not playout_node.is_terminal():
                available_actions = playout_node.get_action_tuples()
                random_action = available_actions[np.random.randint(len(available_actions))]
                playout_node.play_action(random_action)
            adjacency, operations = playout_node.state.operations_and_adjacency()
            model_spec = ModelSpecAPI.ModelSpec(
                    # Adjacency matrix of the module
                    matrix=adjacency,
                    # Operations at the vertices of the module, matches order of matrix
                    ops=operations)
            is_valid = self.api.is_valid(model_spec)
            if i > 1000:
                print(adjacency)
                print(operations)
                print(is_valid)
                break

            # print(f"[PLAYOUT] Playing random action {random_action}")

        reward = self._get_reward(playout_node)

        del playout_node
        return reward


class NASBench101RAVE(NASBench101UCT, RAVE):

    def __init__(self, root_node: NASBench101AMAFNode, api, save_folder=None, params_path=None, disable_tqdm=False):
        super(NASBench101RAVE, self).__init__(root_node, api, save_folder=save_folder, params_path=params_path, disable_tqdm=disable_tqdm)

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

        return NASBench101UCT._playout(self, node)

    def _backpropagation(self, node: Node, result: float):
        """
        Backpropagates the result of a playout up the tree.
        """
        return RAVE._backpropagation(self, node, result)

    def main_loop(self) -> Node:
        """
        Body of UCT
        """
        return NASBench101UCT.main_loop(self)


class NASBench101GRAVE(NASBench101RAVE, GRAVE):

    def __init__(self, root_node: NASBench101AMAFNode, api, save_folder=None, params_path=None, disable_tqdm=False):
        super().__init__(root_node, api, save_folder=save_folder, params_path=params_path, disable_tqdm=disable_tqdm)

    def _selection(self, node: NASBench101AMAFNode) -> NASBench101AMAFNode:
        """
        Selects a candidate child node from the input node.
        """
        return GRAVE._selection(self, node)


class NASBench101NestedMCS(NestedMCS, NASBench101UCT):

    def __init__(self, root_node: NASBench101NestedNode, level, api, save_folder=None, params_path=None,
                 disable_tqdm=False):
        NestedMCS.__init__(self, root_node, level, save_folder=save_folder,
                           disable_tqdm=disable_tqdm, params_path=params_path)
        NASBench101UCT.__init__(self, root_node, api, save_folder, params_path, disable_tqdm)

    def _playout(self, node: NASBench101NestedNode):
        """
        Crée un playout aléatoire et renvoie l'accuracy sur le modèle entraîné
        :return:
        """
        is_valid = False
        i = 0
        while not is_valid:
            i += 1
            node_type = type(node)
            playout_node = node_type(state=copy.deepcopy(node.state), sequence=node.sequence)
            sequence = playout_node.sequence

            while not playout_node.is_terminal():
                available_actions = playout_node.get_action_tuples()
                random_action = available_actions[np.random.randint(len(available_actions))]
                sequence.append(random_action)
                playout_node.play_action(random_action)
            adjacency, operations = playout_node.state.operations_and_adjacency()
            model_spec = ModelSpecAPI.ModelSpec(
                    # Adjacency matrix of the module
                    matrix=adjacency,
                    # Operations at the vertices of the module, matches order of matrix
                    ops=operations)
            is_valid = self.api.is_valid(model_spec)
            if i > 1000:
                print(adjacency)
                print(operations)
                print(is_valid)
                break

        reward = self._get_reward(playout_node)

        del playout_node
        return reward, sequence

    def main_loop(self):
        return NestedMCS.main_loop(self)

    def _score_node(self, child: NASBench101NestedNode, parent: NASBench101NestedNode, C=None) -> float:
        return NASBench101UCT._score_node(self, child, parent, C)


class NASBench101NRPA(NRPA, NASBench101UCT):

    def __init__(self, root_node: NASBench101NestedNode, level, api, save_folder=None, params_path=None,
                 disable_tqdm=False):
        NRPA.__init__(self, root_node, level, save_folder=save_folder,
                      disable_tqdm=disable_tqdm, params_path=params_path)
        NASBench101UCT.__init__(self, root_node, api, save_folder, params_path, disable_tqdm)

    def _score_node(self, child: NASBench101NestedNode, parent: NASBench101NestedNode, C=None) -> float:
        return NASBench101UCT._score_node(self, child, parent, C)

    def _playout(self, node: NASBench101NestedNode):
        is_valid = False
        i = 0
        while not is_valid:
            i += 1
            playout_node = copy.deepcopy(node)
            sequence = playout_node.sequence

            while not playout_node.is_terminal():

                # Vérifier si la policy a une valeur pour ce noeud
                if self._code(playout_node, playout_node.move) not in self.policy:
                    self.policy[self._code(playout_node, playout_node.move)] = 0

                available_actions = playout_node.get_action_tuples()
                probabilities = []
                for move in available_actions:

                    if self._code(playout_node, move) not in self.policy:
                        self.policy[self._code(playout_node, move)] = 0

                policy_values = [self.policy[self._code(playout_node, move)] for move in
                                 available_actions]  # Calcule la probabilité de sélectionner chaque action avec la policy
                probabilities = softmax(policy_values)
                action_index = np.random.choice(np.arange(len(available_actions)), p=probabilities)
                action = available_actions[action_index]  # Used because available_actions is not 1-dimensional

                sequence.append(action)
                playout_node.play_action(action)
                playout_node.hash = playout_node.calculate_zobrist_hash(self.zobrist_table)

            adjacency, operations = playout_node.state.operations_and_adjacency()
            model_spec = ModelSpecAPI.ModelSpec(
                    # Adjacency matrix of the module
                    matrix=adjacency,
                    # Operations at the vertices of the module, matches order of matrix
                    ops=operations)
            is_valid = self.api.is_valid(model_spec)
            if i > 1000:
                print(adjacency)
                print(operations)
                print(is_valid)
                break

        reward = self._get_reward(playout_node)

        del playout_node
        return reward, sequence

class NASBench101UCT_NTK(NASBench101UCT, NTKScorer):

    def __init__(self, root_node: NASBench101Node, api, save_folder=None, params_path=None, disable_tqdm=False):
        super(NASBench101UCT_NTK, self).__init__(root_node, api,
                                                 save_folder=save_folder, params_path=params_path, disable_tqdm=disable_tqdm)

    def read_params(self, path):
        NASBench101UCT.read_params(self, path)
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
        accuracy = NASBench101UCT._get_reward(self, playout_node)

        del playout_node
        return reward, accuracy

    def next_best_move(self, all_rewards=None, best_reward=None, all_accuracies=None, best_accuracy=None) -> Node:
        return NTKScorer.next_best_move(self, all_rewards, best_reward, all_accuracies, best_accuracy)

    def main_loop(self):
        return NTKScorer.main_loop(self)

    def _create_network(self, node: Node) -> nn.Module:
        adjacency, operations = node.state.operations_and_adjacency()
        model_spec = ModelSpecAPI.ModelSpec(
                # Adjacency matrix of the module
                matrix=adjacency,
                # Operations at the vertices of the module, matches order of matrix
                ops=operations)
        is_valid = self.api.is_valid(model_spec)
        assert is_valid
        network = Network((adjacency, operations))
        return network

    def _score_network(self, network):
        return NTKScorer._score_network(self, network, benchmark="nb101")

class NASBench101NRPA_NTK(NASBench101NRPA, NASBench101UCT_NTK, NRPA_NTK):

    def __init__(self, root_node: NASBench101NestedNode, level, api, save_folder=None, params_path=None, disable_tqdm=False):
        NRPA_NTK.__init__(self, root_node, level, save_folder, disable_tqdm, params_path)
        NASBench101NRPA.__init__(self, root_node, level, api, save_folder, params_path, disable_tqdm)
        self.class_for_accuracy = NASBench101NRPA

    def read_params(self, path):
        NASBench101NRPA.read_params(self, path)
        NASBench101UCT_NTK.read_params(self, path)
        NRPA_NTK.read_params(self, path)

    def _playout(self, node: NASBench101NestedNode):
        return NASBench101NRPA._playout(self, node)

    def _get_reward(self, node: Node):
        return NASBench101UCT_NTK._get_reward(self, node)

    def main_loop(self):
        return NRPA_NTK.main_loop(self)

    def _playout(self, node: NASBench101NestedNode):
        is_valid = False
        i = 0
        while not is_valid:
            i += 1
            playout_node = copy.deepcopy(node)
            sequence = playout_node.sequence

            while not playout_node.is_terminal():

                # Vérifier si la policy a une valeur pour ce noeud
                if self._code(playout_node, playout_node.move) not in self.policy:
                    self.policy[self._code(playout_node, playout_node.move)] = 0

                available_actions = playout_node.get_action_tuples()
                probabilities = []
                for move in available_actions:

                    if self._code(playout_node, move) not in self.policy:
                        self.policy[self._code(playout_node, move)] = 0

                policy_values = [self.policy[self._code(playout_node, move)] for move in
                                 available_actions]  # Calcule la probabilité de sélectionner chaque action avec la policy
                probabilities = softmax(policy_values)
                action_index = np.random.choice(np.arange(len(available_actions)), p=probabilities)
                action = available_actions[action_index]  # Used because available_actions is not 1-dimensional

                sequence.append(action)
                playout_node.play_action(action)
                playout_node.hash = playout_node.calculate_zobrist_hash(self.zobrist_table)

            adjacency, operations = playout_node.state.operations_and_adjacency()
            model_spec = ModelSpecAPI.ModelSpec(
                    # Adjacency matrix of the module
                    matrix=adjacency,
                    # Operations at the vertices of the module, matches order of matrix
                    ops=operations)
            is_valid = self.api.is_valid(model_spec)
            if i > 1000:
                print(adjacency)
                print(operations)
                print(is_valid)
                break

        reward = self._get_reward(playout_node)
        info = self.api.query(model_spec)
        test_acc = info['validation_accuracy']

        del playout_node
        return reward, sequence, test_acc

if __name__ == '__main__':
    print("hello")
    node = NASBench101AMAFNode(NASBench101Cell(7))
    print("hellop node")
    mcts = NASBench101RAVE(node, api=None, params_path="/userdata/T0259728/projets/nas/params.json", disable_tqdm=False)