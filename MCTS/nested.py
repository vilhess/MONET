import sys
sys.path.append('/Users/samy/Desktop/NTK4NAS/')

import copy
import json
import shutil
import time
from scipy.special import softmax
import matplotlib.pyplot as plt

import torch
import numpy as np
from tqdm import tqdm
import pandas as pd
from pprint import pprint
import seaborn as sns 
import random

from MCTS import Node
from MCTS.Node import Node, NestedNode
from utils.helpers import running_avg
from MCTS.mcts_agent import MCTSAgent

def softmax_temp(x, tau):
    e_x = np.exp(x / tau)
    return e_x / e_x.sum()

class NestedMCS(MCTSAgent):

    def __init__(self, root_node: NestedNode, level, save_folder=None, disable_tqdm=False, params_path=None):
        super().__init__(root_node, save_folder, params_path=params_path, disable_tqdm=disable_tqdm)
        self.level = level
        self.rewards = []
        self.best_reward = []
        self.best_reward_value = 0

    def _playout(self, node: NestedNode):
        """
        Crée un playout aléatoire et renvoie l'accuracy sur le modèle entraîné
        :return:
        """
        node_type = type(node)
        playout_node = node_type(state=copy.deepcopy(node.state), sequence=copy.deepcopy(node.sequence))
        sequence = playout_node.sequence

        while not playout_node.is_terminal():
            available_actions = playout_node.get_action_tuples()
            random_action = available_actions[np.random.randint(len(available_actions))]
            sequence.append(random_action)
            playout_node.play_action(random_action)

            # print(f"[PLAYOUT] Playing random action {random_action}")

        reward = self._get_reward(playout_node)
        del playout_node
        return reward, sequence

    def nested(self, node, level):

        chosen_sequence = []
        best_sequence = []
        best_score = -1

        while not node.is_terminal():
            action_tuples = node.get_action_tuples()
            # if level == 2: print(f"[LEVEL {level}] Actions: {action_tuples}")
            score_for_state = []
            sequence_for_state = []
            moves_for_state = []

            for action in action_tuples:
                node_type = type(node)
                node_prime = node_type(state=copy.deepcopy(node.state), move=action, parent=node,
                                                sequence=copy.deepcopy(node.sequence))
                node_prime.play_action(action)
                node_prime.sequence.append(action)

                if level == 0:
                    score, sequence = self._playout(node_prime)
                    self.rewards.append(score)
                    if score > self.best_reward_value:
                        self.best_reward_value = score
                    self.best_reward.append(self.best_reward_value)
                else:
                    score, sequence = self.nested(node_prime, level - 1)

                score_for_state.append(score)
                sequence_for_state.append(sequence)
                moves_for_state.append(action)

            high_score = np.max(score_for_state)
            high_index = np.random.choice(
                np.flatnonzero(score_for_state == np.max(score_for_state)))  # Argmax with random tie-breaks
            if high_score >= best_score:
                best_score = high_score
                chosen_move = moves_for_state[high_index]
                best_sequence = sequence_for_state[high_index]
            else:
                try:
                    chosen_move = best_sequence.pop(0)
                except IndexError:
                    print(best_sequence)

            node.play_action(chosen_move)
            # node.sequence.append(chosen_move)
            chosen_sequence.append(chosen_move)

        return best_score, chosen_sequence

    def main_loop(self):
        if self.save_folder is not None:
            shutil.copyfile(self.params_path, f"runs/{self.save_folder}/{self.__class__.__name__}-params.json")
        node = self.root
        reward, sequence = self.nested(node, self.level)
        return node, self.rewards, self.best_reward

class NRPA(NestedMCS):

    def __init__(self, root_node: NestedNode, level, save_folder=None, disable_tqdm=False, params_path=None):
        super().__init__(root_node, level, save_folder, disable_tqdm, params_path)
        self.policy = {}
        # self.n_iter = 100
        # self.alpha = .01  # Learning rate

    def read_params(self, path):
        super().read_params(path)
        with open(path, "r") as f:
            dic = json.load(f)
        self.n_iter = dic["mcts"]["nrpa_n_iter"]
        self.alpha = dic["mcts"]["nrpa_alpha"]
        self.softmax_temp = dic["mcts"]["softmax_temp"]

    def _code(self, node, move):

        if move == None:
            ### SEULEMENT POUR LA RACINE DE L'ARBRE A PRIORI
            return node.hash

        state_code = node.hash
        code = str(state_code) 
        code = ""  # J'enlève le hashage de zobrist pour le moment # justepourvoir
        for i in range(len(move)):
            code = code + str(move[i])

        return code

    def adapt(self, sequence):
        node_type = type(self.root)
        node = node_type(state=copy.deepcopy(self.root.state), move=None, parent=None, sequence=[])

        node.hash = node.calculate_zobrist_hash(self.zobrist_table)
        pol_prime = self.policy.copy()
        for action in sequence:
            code = self._code(node, action)
            # if code not in pol_prime:
            #     print("Erreur 0")
            #     pol_prime[code] = 0
            pol_prime[code] += self.alpha
            z = 0
            moves = node.get_action_tuples()
            for m in moves:
                move_code = self._code(node, m)
                # if move_code not in self.policy:
                #     print("Erreur 1")
                #     self.policy[move_code] = 0
                z += np.exp(self.policy[move_code])
            for m in moves:
                move_code = self._code(node, m)
                # if move_code not in pol_prime:
                #     print("Erreur 2")
                #     pol_prime[move_code] = 0
                pol_prime[move_code] -= self.alpha * (np.exp(self.policy[move_code]) / z)

            node.play_action(action)
            node.hash = node.calculate_zobrist_hash(self.zobrist_table)

        return pol_prime

    def _playout(self, node: NestedNode):
        node_type = type(node)
        playout_node = node_type(state=copy.deepcopy(node.state), move=copy.deepcopy(node.move),
                                 parent=copy.deepcopy(node.parent), sequence=copy.deepcopy(node.sequence))
        sequence = playout_node.sequence
        playout_node.hash = playout_node.calculate_zobrist_hash(self.zobrist_table)

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
                             
            probabilities = softmax_temp(np.array(policy_values), self.softmax_temp)
            # if len(self.best_reward) % 100 == 0:
            #     pprint(list(zip(available_actions, probabilities)))
            action_index = np.random.choice(np.arange(len(available_actions)), p=probabilities)
            action = available_actions[action_index]  # Used because available_actions is not 1-dimensional

            sequence.append(action)
            playout_node.play_action(action)
            playout_node.hash = playout_node.calculate_zobrist_hash(self.zobrist_table)

        reward = self._get_reward(playout_node)

        del playout_node
        return reward, sequence

    def nrpa(self, node, level):

        if level == 0:
            self.pbar.update(1)
            self.pbar.set_description(f"Current best reward : {self.best_reward_value:.4f}")
            # if (len(self.rewards)+1) % 1000 == 0:
            #     f, ax = plt.subplots(1,1)
            #     ax.plot(running_avg(self.rewards, 10))
            #     f.savefig("rewards.png")
            #     plt.close(f)
            #     f, ax = plt.subplots(1,1)
            #     ax.plot(running_avg(self.best_reward, 10))
            #     f.savefig("best_rewards.png")
            #     plt.close(f)
            #     node_type = type(self.root)
            #     playout_node = node_type(copy.deepcopy(self.root.state), sequence=[])
            #     playout_node.hash = playout_node.calculate_zobrist_hash(self.zobrist_table)

            #     act = playout_node.get_action_tuples()
            #     probas = softmax_temp(np.array([self.policy[self._code(playout_node, move)] for move in act]), self.softmax_temp)

            #     f, ax = plt.subplots(1,1)
            #     df = pd.DataFrame({"action": act, "probability": probas}).sort_values('probability', ascending=False)
            #     sns.barplot(df.iloc[:20], x="action", y="probability", color="#3d405b")
            #     plt.xticks(rotation=90)
            #     f.savefig("best_actions.png")
            #     plt.close(f)
            #     print(f"[{len(self.rewards)}/{self.n_iter ** self.level}] Best reward: {self.best_reward_value}")
            
            score, sequence = self._playout(self.root)
            self.rewards.append(score)
            if score > self.best_reward_value:
                self.best_reward_value = score
                # print(f"[{len(self.rewards)}/{self.n_iter ** self.level}] Best reward: {self.best_reward_value}")

            self.best_reward.append(self.best_reward_value)
            return score, sequence

        else:
            # For nesting levels >= 1
            best_score = -1
            best_sequence = []
            ## BREAK CONDITION
            # if len(self.rewards) > 5000:
            #     return best_score, best_sequence

            for i in range(self.n_iter):
                t1 = time.time()
                reward, sequence = self.nrpa(node, level - 1)
                t2 = time.time()
                # if level == 2:
                #     print(f"NRPA search at level {level - 1} has taken {(t2-t1):.2f} seconds")
                if reward >= best_score:
                    best_score = reward
                    best_sequence = sequence
                self.policy = self.adapt(best_sequence)

            return best_score, best_sequence

    def main_loop(self):
        node = self.root
        pol = {}
        self.pbar = tqdm(total=self.n_iter ** self.level, position=0, leave=True)
        t1 = time.time()
        reward, sequence = self.nrpa(node, self.level)
        t2 = time.time()
        self.pbar.close()
        print(f"Sequence is {sequence} with score {reward}")
        for action in sequence:
            node.play_action(action)
        return node, self.rewards, self.best_reward

class NestedMCS_NTK(NestedMCS):

    def __init__(self, root_node: NestedNode, level, save_folder=None, disable_tqdm=False, params_path=None):
        super().__init__(root_node, level, save_folder, params_path=params_path, disable_tqdm=disable_tqdm)
        self.accuracies = []
        self.best_accuracy = []
        self.best_accuracy_value = 0
        self.class_for_accuracy = None  # La classe qu'on invoque pour obtenir l'accuraacy

    def _playout(self, node: NestedNode):
        """
        Crée un playout aléatoire et renvoie l'accuracy sur le modèle entraîné
        :return:
        """
        node_type = type(node)
        playout_node = node_type(state=copy.deepcopy(node.state), sequence=copy.deepcopy(node.sequence))
        sequence = playout_node.sequence

        while not playout_node.is_terminal():
            available_actions = playout_node.get_action_tuples()
            random_action = available_actions[np.random.randint(len(available_actions))]
            sequence.append(random_action)
            playout_node.play_action(random_action)

            # print(f"[PLAYOUT] Playing random action {random_action}")

        reward = self._get_reward(playout_node)
        accuracy = self.class_for_accuracy._get_reward(self, playout_node)
        # print(reward, accuracy)
        del playout_node
        return reward, sequence, accuracy

    def nested(self, node, level):

        """
        La seule chose qui change est l'ajout de nouvelles métriques pour tracker l'accuracy
        :param node:
        :param level:
        :return:
        """

        chosen_sequence = []
        best_sequence = []
        best_score = -1

        while not node.is_terminal():
            action_tuples = node.get_action_tuples()
            # if level == 2: print(f"[LEVEL {level}] Actions: {action_tuples}")
            score_for_state = []
            sequence_for_state = []
            moves_for_state = []

            for action in action_tuples:
                node_type = type(node)
                node_prime = node_type(state=copy.deepcopy(node.state), move=action, parent=node,
                                                sequence=copy.deepcopy(node.sequence))
                node_prime.play_action(action)
                node_prime.sequence.append(action)

                if level == 0:
                    score, sequence, accuracy = self._playout(node_prime)
                    self.rewards.append(score)
                    self.accuracies.append(accuracy)
                    if score > self.best_reward_value:
                        self.best_reward_value = score
                        self.best_accuracy_value = accuracy
                    self.best_reward.append(self.best_reward_value)
                    self.best_accuracy.append(self.best_accuracy_value)

                else:
                    score, sequence = self.nested(node_prime, level - 1)

                score_for_state.append(score)
                sequence_for_state.append(sequence)
                moves_for_state.append(action)

            high_score = np.max(score_for_state)
            high_index = np.random.choice(
                np.flatnonzero(score_for_state == np.max(score_for_state)))  # Argmax with random tie-breaks
            if high_score >= best_score:
                best_score = high_score
                chosen_move = moves_for_state[high_index]
                best_sequence = sequence_for_state[high_index]
            else:
                try:
                    chosen_move = best_sequence.pop(0)
                except IndexError:
                    print(best_sequence)

            node.play_action(chosen_move)
            # node.sequence.append(chosen_move)
            chosen_sequence.append(chosen_move)

        return best_score, chosen_sequence

    def main_loop(self):
        if self.save_folder is not None:
            shutil.copyfile(self.params_path, f"runs/{self.save_folder}/{self.__class__.__name__}-params.json")
        node = self.root

        reward, sequence = self.nested(node, self.level)
        return node, self.rewards, self.best_reward, self.accuracies, self.best_accuracy

class NRPA_NTK(NRPA):

    def __init__(self, root_node: NestedNode, level, save_folder=None, disable_tqdm=False, params_path=None):
        super().__init__(root_node, level, save_folder, disable_tqdm, params_path)
        self.accuracies = []
        self.best_accuracy = []
        self.best_accuracy_value = 0
        self.class_for_accuracy = None  # La classe qu'on invoque pour calculer l'accuracy


    def _playout(self, node: NestedNode):

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
            probabilities = softmax_temp(np.array(policy_values), self.softmax_temp)
            action_index = np.random.choice(np.arange(len(available_actions)), p=probabilities)
            action = available_actions[action_index]  # Used because available_actions is not 1-dimensional

            sequence.append(action)
            playout_node.play_action(action)
            playout_node.hash = playout_node.calculate_zobrist_hash(self.zobrist_table)

        reward = self._get_reward(playout_node)
        accuracy = self.class_for_accuracy._get_reward(self, playout_node)

        # print(reward, accuracy)
        del playout_node
        return reward, sequence, accuracy

    def nrpa(self, node, level):
        """
        La seule chose qui change est l'ajout des nouvelles métriques
        :param node:
        :param level:
        :return:
        """

        if level == 0:
            self.pbar.update(1)
            self.pbar.set_description(f"Current best reward : {self.best_reward_value:.4f}, best accuracy : {self.best_accuracy_value:.4f}")
     
            score, sequence, accuracy = self._playout(self.root)

            self.rewards.append(score)
            self.accuracies.append(accuracy)
            if score > self.best_reward_value:
                self.best_reward_value = score
                self.best_accuracy_value = accuracy
            self.best_reward.append(self.best_reward_value)
            self.best_accuracy.append(self.best_accuracy_value)
            return score, sequence

        else:
            # For nesting levels >= 1
            best_score = -1
            best_sequence = []

            for i in range(self.n_iter):
                reward, sequence = self.nrpa(node, level - 1)

                if reward >= best_score:
                    best_score = reward
                    best_sequence = sequence
                self.policy = self.adapt(best_sequence)

            return best_score, best_sequence

    def main_loop(self):
        node = self.root
        pol = {}
        t1 = time.time()
        self.pbar = tqdm(total=self.n_iter ** self.level, position=0, leave=True)

        reward, sequence = self.nrpa(node, self.level)
        t2 = time.time()
        self.pbar.close()

        print(f"Sequence is {sequence} with score {reward}")
        for action in sequence:
            node.play_action(action)
        return node, self.rewards, self.best_reward, self.accuracies, self.best_accuracy
