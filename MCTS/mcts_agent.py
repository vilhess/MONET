import copy
import json
import pickle
import shutil
from datetime import datetime
import os
import sys
from torch import nn

from ntk.compute_score import compute_score
from utils.CIFAR import CIFAR100Dataset, CIFAR10Dataset

sys.path.append("..")


import time
import itertools
from pprint import pprint

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.losses import TverskyLoss
from utils.helpers import create_now_folder, subset_classes
from utils.model_trainer import ModelTrainer
from utils.pytorch_dataset import RadarDataset, RadarDavaDataset

from .Node import Node, AMAFNode


class MCTSAgent:

    def __init__(self, root_node: Node, trainer: ModelTrainer=None, save_folder=None, params_path=None, disable_tqdm=False):
        self.playouts_per_selection = None
        self.root = root_node
        self.trainer = trainer
        self.playout_epochs = None
        self.C = None
        self.n_iter = None
        self.max_params = None
        self.save_folder = save_folder
        self.search_time = None
        self.disable_tqdm = disable_tqdm
        if params_path is not None:
            self.params_path = params_path
        elif params_path is None and trainer is not None:
            self.params_path = trainer.params_path
        self.read_params(self.params_path)


    def read_params(self, path):
        with open(path, "r") as f:
            dic = json.load(f)
        self.playout_epochs = dic["mcts"]["playout_epochs"]
        self.C = dic["mcts"]["C"]
        self.n_iter = dic["mcts"]["n_iter"]
        self.max_params = dic["mcts"]["max_params"]
        self.playouts_per_selection = dic["mcts"]["playouts_per_selection"]
        search_time = dic["mcts"]["search_time"]
        if search_time > 0:
            self.search_time = search_time

    def _score_node(self, node: Node, parent: Node):
        pass

    def _get_reward(self, node: Node):
        pass

    def _create_network(self, node: Node) -> nn.Module:
        """
        Créer un réseau de neurones à partir du noeud MCTS
        :param node:
        :return: nn.Module
        """
        pass

    def _selection(self, node: Node):
        pass

    def _expansion(self, node: Node):
        pass

    def _playout(self, node: Node):
        pass

    def _backpropagation(self, node: Node, result: float):
        pass

    def __call__(self):
        pass

class NTKScorer(MCTSAgent):

    def __init__(self, root_node: Node, trainer: ModelTrainer=None, save_folder=None, params_path=None, disable_tqdm=False):
        self.train_loader = None
        self.device = None
        super().__init__(root_node, params_path=params_path, save_folder=save_folder, disable_tqdm=disable_tqdm, trainer=trainer)
        dataset = CIFAR10Dataset()
        self.dataset_classes, self.class_permutation = subset_classes(dataset=dataset, samples_per_class=10, device=self.device,
                                                            subsample=10)

    def read_params(self, path):
        super().read_params(path)
        with open(path, "r") as f:
            dic = json.load(f)
        # PATH = dic["general"]["data_path"]
        # dataset = RadarDavaDataset(root_dir=PATH, has_distance=False, batch_size=2)
        # self.train_loader, test_loader, val_loader = dataset.generate_loaders(test_split=.8, val_split=.8)
        self.device = torch.device(dic["hardware"]["device"])

    def _score_network(self, network: nn.Module, benchmark="nb101"):
        # assert self.trainer is not None, "We need a ModelTrainer object to fetch a batch of data"
        # assert self.trainer.train_data is not None, "ModelTrainer train_data is None"

        network.to(self.device)
        # x = next(iter(self.train_loader))[0].to(self.device)
        score = compute_score(network, self.dataset_classes, class_permutation=self.class_permutation, device=self.device, benchmark=benchmark)
        return score

    def next_best_move(self, all_rewards=None, best_reward=None, all_accuracies=None, best_accuracy=None) -> Node:
        """
        On modifie uniquement les ajouts de valeurs dans all_accuracies et best_accuracy
        """
        best_reward_value = np.max(best_reward) if len(best_reward) > 0 else 0
        best_acc = np.max(best_accuracy) if len(best_accuracy) > 0 else 0
        t0 = time.time()
        for i in tqdm(range(self.n_iter), disable=self.disable_tqdm):

            # region Timed search
            t = time.time()
            if t - t0 > self.search_time:
                print("Time's up !")
                break
            # endregion

            leaf_node = self._selection(self.root)
            expanded_node = self._expansion(leaf_node)

            for i_playout in range(self.playouts_per_selection):
                result, accuracy = self._playout(expanded_node)
                _ = self._backpropagation(expanded_node, result)
                all_rewards.append(result)
                all_accuracies.append(accuracy)
                if result > best_reward_value:
                    best_reward_value = result
                    best_acc = accuracy
                best_reward.append(best_reward_value)
                best_accuracy.append(best_acc)

        best_move_id = np.argmax([np.mean(child.results) for child in self.root.get_children()])
        best_move = self.root.get_children()[best_move_id]
        # print(f"[BODY] Selecting best move {best_move.move} with mean result {np.mean(best_move.results)}")

        return best_move, all_rewards, best_reward, all_accuracies, best_accuracy

    def main_loop(self):
        """
        Corps de l'algorithme. Cherche le meilleur prochain coup jusqu'à avoir atteint un état terminal.
        :return: Le noeud représentant les meilleurs coups.
        """
        """Enregistrer les paramètres de la simul ation dans le folder"""
        # if self.save_folder is not None:
        #     shutil.copyfile(self.params_path, f"runs/{self.save_folder}/{self.__class__.__name__}-params.json")
        node = self.root
        all_rewards = []
        best_reward = []
        all_accuracies = []
        best_accuracy = []
        best_reward_value = 0

        while not node.is_terminal():
            best_move, all_rewards, best_reward, all_accuracies, best_accuracies = self.next_best_move(all_rewards,
                                                                                                       best_reward,
                                                                                                       all_accuracies,
                                                                                                       best_accuracy)
            node.play_action(best_move.move)
            root_type = type(self.root)
            self.root = root_type(node.state)

        return node, all_rewards, best_reward, all_accuracies, best_accuracy



class UCT(MCTSAgent):

    def __init__(self, root_node: Node, save_folder=None, disable_tqdm=False, trainer: ModelTrainer=None, params_path=None):
        super().__init__(root_node, params_path=params_path, save_folder=save_folder, disable_tqdm=disable_tqdm, trainer=trainer)

    def _score_node(self, child: Node, parent: Node, C=None) -> float:
        # Returns UCB score for a child node
        if len(child.results) == 0:  # Si le noeud n'a pas encore été visité !
            return np.inf
        if C is None:
            C = self.C

        mu_i = np.mean(child.results)
        # print(f"[UCB] : move : {child.move}, mu_i = {mu_i}, autre param: {C * (np.sqrt(np.log(len(parent.results)) / len(child.results)))}")
        return mu_i + C * (np.sqrt(np.log(len(parent.results)) / len(child.results)))

    def _selection(self, node: Node) -> Node:
        """
        Selects a candidate child node from the input node.
        """
        if not node.is_leaf():  # Tant que l'on a pas atteint une feuille de l'arbre
            # Choisir le meilleur noeud enfant
            # print(f"[SELECTION] nb of candidates : {len(node.get_children())}")
            # C dynamique
            C_dif = np.max([np.nan_to_num(np.mean(c.results)) for c in node.get_children()]) - np.min(
                [np.nan_to_num(np.mean(c.results)) for c in node.get_children()])
            # print(f"[SELECTION] C_dif = {C_dif}")
            C = max([self.C, 1 * C_dif])
            # print(f"[SELECTION] Exploration constant C : {C}")
            scores = [self._score_node(child, node, C) for child in node.get_children()]
            candidate_id = np.random.choice(np.flatnonzero(scores == np.max(scores)))  # Argmax with random tie-breaks
            candidate = node.get_children()[candidate_id]
            # print(f"[SELECTION] Choosing child {candidate.move} with ucb {self._score_node(candidate, node, C)}")
            return self._selection(candidate)

        # self.current_game_moves.append(node.move)
        return node

    def _expansion(self, node: Node) -> Node:
        """
        Unless L ends the game decisively (e.g. win/loss/draw) for either player,
        create one (or more) child nodes and choose node C from one of them.
        Child nodes are any valid moves from the game position defined by L.
        """
        if not node.is_terminal():
            """
            Si le noeud n'a pas encore été exploré : on le retourne directement
            """
            if len(node.results) == 0 and node.parent is not None:
                return node
            node_type = type(node)
            node.children = [node_type(copy.deepcopy(node.state),
                                       move=m,
                                       parent=node)
                             for m in node.get_action_tuples()]
            # pprint(node.get_action_tuples())

            # Play the move for each child (updates the board in the child nodes)
            for child in node.children:
                child.play_action(child.move)
            returned_node = node.children[np.random.randint(0, len(node.children))]
            # print(f"[EXPANSION] returning random child : {returned_node.move}")
            return returned_node

        return node

    def _get_reward(self, node: Node):
        network = self._create_network(node)
        n_params = sum(p.numel() for p in network.parameters())
        self.trainer.set_model(network)
        self.trainer.set_parameters(self.trainer.params_path)
        self.trainer.n_epochs = self.playout_epochs
        self.trainer.train_model()
        reward = 1 - np.mean(self.trainer.evaluate()["loss"])
        del network
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

        del playout_node
        return reward

    def _backpropagation(self, node: Node, result: float):
        """
        Backpropagates the result of a playout up the tree.
        """
        if node.parent is None:
            node.results.append(result)
            return "Done"
        node.results.append(result)  # Ajouter le résultat à la liste
        return self._backpropagation(node.parent, result)  # Fonction récursive

    def next_best_move(self, all_rewards=None, best_reward=None) -> Node:
        """
        Body of UCT
        """
        best_reward_value = np.max(best_reward) if len(best_reward) > 0 else 0
        t0 = time.time()
        for i in tqdm(range(self.n_iter), disable=self.disable_tqdm):

            # region Timed search
            t = time.time()
            if t - t0 > self.search_time:
                print("Time's up !")
                break
            # endregion

            leaf_node = self._selection(self.root)
            expanded_node = self._expansion(leaf_node)

            for i_playout in range(self.playouts_per_selection):
                result = self._playout(expanded_node)
                _ = self._backpropagation(expanded_node, result)
                all_rewards.append(result)
                if result > best_reward_value:
                    best_reward_value = result
                best_reward.append(best_reward_value)

        best_move_id = np.argmax([np.mean(child.results) for child in self.root.get_children()])
        best_move = self.root.get_children()[best_move_id]
        # print(f"[BODY] Selecting best move {best_move.move} with mean result {np.mean(best_move.results)}")

        return best_move, all_rewards, best_reward

    def main_loop(self):
        """
        Corps de l'algorithme. Cherche le meilleur prochain coup jusqu'à avoir atteint un état terminal.
        :return: Le noeud représentant les meilleurs coups.
        """
        """Enregistrer les paramètres de la simul ation dans le folder"""
        # if self.save_folder is not None:
        #     shutil.copyfile(self.params_path, f"runs/{self.save_folder}/{self.__class__.__name__}-params.json")
        node = self.root
        self.all_rewards = []
        self.best_reward = []
        self.best_reward_value = 0

        while not node.is_terminal():
            best_move, self.all_rewards, self.best_reward = self.next_best_move(self.all_rewards, self.best_reward)
            print(best_move.move)

            node.play_action(best_move.move)
            root_type = type(self.root)
            self.root = best_move
            # print(len(best_move.children))
            # print([(len(e.results), np.mean(e.results)) for e in best_move.children])
            # self.root = root_type(copy.deepcopy(node.state))

        return node, self.all_rewards, self.best_reward

class RAVE(UCT):

    def __init__(self, root_node: Node, params_path=None, save_folder=None, disable_tqdm=False, trainer: ModelTrainer=None):
        self.list_nodes = []
        self.list_nodes.append(root_node)
        self.b = None
        super(RAVE, self).__init__(root_node, params_path=params_path, save_folder=save_folder, disable_tqdm=disable_tqdm, trainer=trainer)

    def read_params(self, path):
        super().read_params(path)
        with open(path, "r") as f:
            dic = json.load(f)
        self.b = dic["mcts"]["rave_b"]

    def beta(self, ni, ni_tilda):
        """
        D'après Gelly et Silver, beta(ni, ni_tilda) = n_tilda / (ni + ni_tilda + 4b^2ni*ni_tilda)
        :param ni: Nombre de parties du noeud i
        :param ni_tilda: Nombre de parties contenant le noeud i
        :return:
        """
        p = ni_tilda
        d = ni + ni_tilda + (4 * np.power(self.b, 2) * ni * ni_tilda)
        return p / d

    def _score_node(self, child: Node, parent: Node, C=None) -> int:
        # Returns UCB score for a child node
        if len(child.results) == 0:  # Si le noeud n'a pas encore été visité !
            return np.inf
        if C is None:
            C = self.C
        mu_i = np.mean(child.results)
        mu_i_tilda = np.nan_to_num(np.mean(child.amaf), 0)
        beta = self.beta(ni=len(child.results),
                         ni_tilda=len(child.amaf))
        exploration_term = C * (np.sqrt(np.log(len(parent.results)) / len(child.results)))
        # print(f"[UCB RAVE] : move : {child.move}, mu_i = {mu_i}, mu_i_tilda = {mu_i_tilda}, beta = {beta}, autre param: {C * (np.sqrt(np.log(len(parent.results)) / len(child.results)))}")

        return (1 - beta) * mu_i + beta * mu_i_tilda + exploration_term

    """
    Pas besoin de redéfinir la méthode de sélection
    """

    def _expansion(self, node: AMAFNode) -> AMAFNode:
        """
        Unless L ends the game decisively (e.g. win/loss/draw) for either player,
        create one (or more) child nodes and choose node C from one of them.
        Child nodes are any valid moves from the game position defined by L.
        """
        if not node.is_terminal():
            """
            Si le noeud n'a pas encore été exploré : on le retourne directement
            """
            if len(node.results) == 0 and node.parent is not None:
                return node
            node_type = type(node)
            node.children = [node_type(copy.deepcopy(node.state),
                                       move=m,
                                       parent=node)
                             for m in node.get_action_tuples()]
            # pprint(node.get_action_tuples())

            # Play the move for each child (updates the board in the child nodes)
            for child in node.children:
                self.list_nodes.append(child)
                child.play_action(child.move)
            returned_node = node.children[np.random.randint(0, len(node.children))]
            # print(f"[EXPANSION] returning random child : {returned_node.move}")
            return returned_node

        return node

    """
    Pas besoin de redéfinir la méthode de playout
    """

    def _backpropagation(self, node: AMAFNode, result: float):
        """
        Backpropagates the result of a playout up the tree.
        Also backpropagates the AMAF results.
        :param node: Current node
        :param result: Result of the playout
        :return:
        """
        if node.parent is None:
            node.results.append(result)
            return "Done"

        node.results.append(result)  # Ajouter le résultat à la liste
        for temp_node in self.list_nodes:
            if node.move == temp_node.move :#and temp_node.has_predecessor(node):
                temp_node.amaf.append(result)
        return self._backpropagation(node.parent, result)  # Fonction récursive

class TrainingFreeRAVE(RAVE):

    def __init__(self, root_node: Node, trainer: ModelTrainer, save_folder=None, disable_tqdm=False):
        super().__init__(root_node, trainer, save_folder, disable_tqdm)
        self.metric = NASWOT(trainer)

    def _score_network(self, network):
        self.trainer.set_model(network)
        self.trainer.set_parameters(self.trainer.params_path)
        self.metric.reset()
        score = self.metric.score(network)
        n_params = sum(p.numel() for p in network.parameters())
        # penalty = 1 if n_params - self.max_params < 0 else n_params - self.max_params
        penalty = 0 if n_params - self.max_params < 0 else -(1-(n_params/120441))**2
        print(f"Score is {score}, penalty is {penalty}, so score = {score + penalty}")
        score += penalty

        return score

class GRAVE(RAVE):

    def __init__(self, root_node: Node, params_path=None, save_folder=None, disable_tqdm=False, trainer: ModelTrainer = None):
        self.ref = None
        super(GRAVE, self).__init__(root_node, params_path=params_path, save_folder=save_folder, disable_tqdm=disable_tqdm, trainer=trainer)

    def read_params(self, path):
        super().read_params(path)
        with open(path, "r") as f:
            dic = json.load(f)
        self.ref = dic["mcts"]["grave_ref"]


    def _score_node(self, child: AMAFNode, parent: Node, C=None) -> int:
        # Returns UCB score for a child node
        if len(child.results) == 0:  # Si le noeud n'a pas encore été visité !
            return np.inf
        if C is None:
            C = self.C
        ref_node = child
        i=0
        while len(ref_node.results) < self.ref:
            if ref_node.parent.parent is None:
                break
            if i > 100:
                break
            ref_node = child.parent
            i +=1

        mu_i = np.mean(child.results)
        mu_i_tilda = np.nan_to_num(np.mean(ref_node.amaf), 0)
        beta = self.beta(ni=len(child.results),
                         ni_tilda=len(ref_node.amaf))
        exploration_term = C * (np.sqrt(np.log(len(parent.results)) / len(child.results)))
        # print(f"[UCB GRAVE] : move : {child.move}, mu_i = {mu_i}, mu_i_tilda = {mu_i_tilda}, beta = {beta}, autre param: {C * (np.sqrt(np.log(len(parent.results)) / len(child.results)))}")
        return (1 - beta) * mu_i + beta * mu_i_tilda + exploration_term

class TrainingFreeGRAVE(GRAVE):

    def __init__(self, root_node: Node, trainer: ModelTrainer, save_folder=None, disable_tqdm=False):
        super().__init__(root_node, trainer, save_folder, disable_tqdm)
        self.metric = NASWOT(trainer)

    def _score_network(self, network):
        self.trainer.set_model(network)
        self.trainer.set_parameters(self.trainer.params_path)
        self.metric.reset()
        score = self.metric.score(network)
        return score

