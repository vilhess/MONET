import sys
sys.path.append('/Users/samy/Desktop/NTK4NAS/')

import os
from collections import namedtuple

from search_spaces.DARTS.search_cnn import NetworkCIFAR
from MCTS.mcts_agent import UCT, NTKScorer
from MCTS.nested import *
from MCTS.Node import Node
from search_spaces.nas_bench_301.NASBench301Node import DARTSNode, DARTSAMAFNode, DARTSNestedNode
import nasbench301 as nb
import numpy as np
from torch import nn

N_NODES = 6
ADJACENCY_MATRIX_SIZE = N_NODES ** 2
N_OPERATIONS = 7

class NASBench301UCT(UCT):

    def __init__(self, root_node: DARTSNode, performance_model=None, save_folder=None, disable_tqdm=False, params_path=None):
        super().__init__(root_node, save_folder=save_folder, disable_tqdm=disable_tqdm, params_path=params_path)
        if performance_model is None: 
            models_1_0_dir = "API/nb_models"
            model_paths = {
                model_name: os.path.join(models_1_0_dir, '{}_v1.0'.format(model_name))
                for model_name in ['xgb', 'lgb_runtime']
            }
            self.performance_model = nb.load_ensemble(model_paths['xgb'])
        else:
            self.performance_model = performance_model
        self.zobrist_table = []
        for _ in range(2):  # Une fois pour la normal cell et une fois pour la reduction cell
            for i in range(ADJACENCY_MATRIX_SIZE):
                adjacency_table = []
                for operation in range(N_OPERATIONS):
                    adjacency_table.append(random.randint(0, 2 ** 64))
                self.zobrist_table.append(adjacency_table)
        
        self.hash_table = {}
        self.root.add_zobrist(self.zobrist_table)

    def _get_reward(self, node: DARTSNode):
        normal_cell_genotype = node.state[0].to_genotype()
        reduction_cell_genotype = node.state[1].to_genotype()  # REMPLACER AVEC 1, C'EST JUSTE POUR LE CALCUL DU NTK
        Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')
        genotype_config = Genotype(
                normal=normal_cell_genotype,
                normal_concat=[2, 3, 4, 5],
                reduce=reduction_cell_genotype,
                reduce_concat=[2, 3, 4, 5]
        )
        accuracy_prediction_genotype = self.performance_model.predict(config=genotype_config, representation="genotype",
                                                        with_noise=True)

        return accuracy_prediction_genotype


class NASBench301UCT_NTK(NASBench301UCT, NTKScorer):

        def __init__(self, root_node: DARTSNode, performance_model=None, save_folder=None, params_path=None, disable_tqdm=False):
            super(NASBench301UCT_NTK, self).__init__(root_node, performance_model=performance_model, save_folder=save_folder, params_path=params_path,
                                                     disable_tqdm=disable_tqdm)

        def read_params(self, path):
            NASBench301UCT.read_params(self, path)
            NTKScorer.read_params(self, path)

        def _get_reward(self, node: DARTSNode):
            network = self._create_network(node)
            reward = self._score_network(network)
            return reward

        def _playout(self, node: DARTSNode):
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
            accuracy = NASBench301UCT._get_reward(self, playout_node)

            del playout_node
            return reward, accuracy

        def _create_network(self, node: DARTSNode) -> nn.Module:
            normal_cell_genotype = node.state[0].to_genotype()
            reduction_cell_genotype = node.state[1].to_genotype()  # REMPLACER AVEC 1, C'EST JUSTE POUR LE CALCUL DU NTK
            Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')
            genotype_config = Genotype(
                    normal=normal_cell_genotype,
                    normal_concat=[2, 3, 4, 5],
                    reduce=reduction_cell_genotype,
                    reduce_concat=[2, 3, 4, 5]
            )
            network = NetworkCIFAR(3, 10, 5, False, genotype_config).to(self.device)
            return network

        def _score_network(self, network):
            return NTKScorer._score_network(self, network)

        def next_best_move(self, all_rewards=None, best_reward=None, all_accuracies=None, best_accuracy=None) -> DARTSNode:
            return NTKScorer.next_best_move(self, all_rewards, best_reward, all_accuracies, best_accuracy)

        def main_loop(self):
            return NTKScorer.main_loop(self)


class NASBench301NestedMCS(NestedMCS, NASBench301UCT):

    def __init__(self, root_node: DARTSNestedNode, level, performance_model=None, save_folder=None, params_path=None,
                 disable_tqdm=False):
        NestedMCS.__init__(self, root_node, level, save_folder=save_folder,
                           disable_tqdm=disable_tqdm, params_path=params_path)
        NASBench301UCT.__init__(self, root_node, performance_model, save_folder, disable_tqdm, params_path)

    def main_loop(self):
        return NestedMCS.main_loop(self)

    def _get_reward(self, node: DARTSNestedNode):
        return NASBench301UCT._get_reward(self, node)

class NASBench301NRPA(NRPA, NASBench301UCT):

    def __init__(self, root_node: DARTSNestedNode, level, save_folder=None, params_path=None,
                 disable_tqdm=False):
        NRPA.__init__(self, root_node, level, save_folder=save_folder,
                      disable_tqdm=disable_tqdm, params_path=params_path)
        NASBench301UCT.__init__(self, root_node, save_folder, disable_tqdm, params_path)

    def _get_reward(self, node: DARTSNode):
        return NASBench301UCT._get_reward(self, node)
        

class NASBench301NestedMCS_NTK(NASBench301NestedMCS, NASBench301UCT_NTK, NestedMCS_NTK):

    def __init__(self, root_node: DARTSNestedNode, level, performance_model=None, save_folder=None, params_path=None,
                 disable_tqdm=False):
        NestedMCS_NTK.__init__(self, root_node, level, save_folder, disable_tqdm, params_path)
        NASBench301NestedMCS.__init__(self, root_node, level, performance_model, save_folder, params_path, disable_tqdm)
        self.class_for_accuracy = NASBench301NestedMCS

    def _get_reward(self, node: Node):
        return np.random.rand()
        return NASBench301UCT_NTK._get_reward(self, node)

    def _playout(self, node: Node):
        return NestedMCS_NTK._playout(self, node)

    def main_loop(self):
        return NestedMCS_NTK.main_loop(self)

class NASBench301NRPA_NTK(NASBench301NRPA, NASBench301UCT_NTK, NRPA_NTK):

    def __init__(self, root_node: DARTSNestedNode, level, save_folder=None, params_path=None,
                 disable_tqdm=False):
        NRPA_NTK.__init__(self, root_node, level, save_folder, disable_tqdm, params_path)
        NASBench301NRPA.__init__(self, root_node, level, save_folder=save_folder,
                                 disable_tqdm=disable_tqdm, params_path=params_path)
        self.class_for_accuracy = NASBench301NRPA

    def read_params(self, path):
        NASBench301NRPA.read_params(self, path)
        NASBench301UCT_NTK.read_params(self, path)
        NRPA_NTK.read_params(self, path)

    def _playout(self, node: DARTSNestedNode):
        return NRPA_NTK._playout(self, node)

    def _get_reward(self, node: Node):
        return NASBench301UCT_NTK._get_reward(self, node)

    def main_loop(self):
        return NRPA_NTK.main_loop(self)