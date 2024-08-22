from itertools import chain
from typing import Tuple
import sys


class Node:
    """
    Représente un noeud dans l'arbre de recherche.
    """

    def __init__(self, state, move=None, parent=None):
        self.state = state
        self.move = move
        self.parent = parent
        self.children = []
        self.results = []

    def get_children(self):
        return self.children

    def is_leaf(self):
        return True if len(self.children) == 0 else False

    def is_terminal(self):
        return self.state.is_complete()

    def get_action_tuples(self):
        pass

    def play_action(self, action):
        pass

    def has_predecessor(self, node):
        """
        Return True if self is a child of node (can be several generations)
        :param node:
        :return:
        """
        temp_parent = self.parent
        while temp_parent.parent is not None:
            if temp_parent == node:
                return True
            temp_parent = temp_parent.parent
        if temp_parent == node:  # Include root (pas sûr)
            return True
        return False


class AMAFNode(Node):

    def __init__(self, state, move=None, parent=None):
        super().__init__(state, move, parent)
        self.amaf = []


class NestedNode(Node):

    def __init__(self, state, move=None, parent=None, sequence=None):
        super().__init__(state, move, parent)
        self.sequence = sequence

# def get_network_action_tuples(list_keys):
#     tuples = []
#     if "n_floors" in list_keys:
#         for i in [1,2,3,4]:
#             tuples.append(("n_floors", i))
#     if "n_repeats" in list_keys:
#         for i in [1,2,3,4]:
#             tuples.append(("n_repeats", i))
#     if "first_step_channels" in list_keys:
#         for i in [4,8,12,16,20,32]:
#             tuples.append(("first_step_channels", i))
#     return tuples
#
# class NASNetNode(Node):
#     """
#     Chercher les 3 cellules NASNet simultanément.
#     """
#
#     def __init__(self, state: Tuple[Cell, Cell, Cell], move=None, parent=None, state_dict=None, network_params=None):
#         super(NASNetNode, self).__init__(state, move, parent, state_dict)
#         self.network_params = {"n_floors": None, "n_repeats": None, "first_step_channels": None}
#         if network_params is not None:
#             self.network_params = network_params
#
#
#     def get_action_tuples(self):
#         list_tuples = []
#         for i, s in enumerate(self.state):  # Itérer sur les trois cellules
#             list_tuples.append([(i, *t) for t in s.get_action_tuples()])
#         # region Network structure parameters
#         network_params = [k for k, v in self.network_params.items() if v is None]
#         list_tuples.append(get_network_action_tuples(network_params))
#         # endregion
#         tuples = list(chain.from_iterable(list_tuples))  # Pour flatten la liste
#         return tuples
#
#     def is_terminal(self):
#         return all([c.is_complete() for c in self.state]) and None not in list(self.network_params.values())
#
#     def play_action(self, action):
#         """
#         Jouer une action (i.e. choisir un input, opération ou combinaison)
#         :param action: un tuple (cell, block, action, action_choice)
#         :return:
#         """
#         if len(action) == 4:  # (cell, block, action, action_choice)
#             self.state[action[0]].play_action(*action[1:])
#         elif len(action) == 2:
#             self.network_params[action[0]] = action[1]
#
#
# class NASNetAMAFNode(NASNetNode):
#
#     def __init__(self, state: Tuple[Cell, Cell, Cell], move=None, parent=None, state_dict=None, network_params=None):
#         super(NASNetAMAFNode, self).__init__(state, move, parent, state_dict, network_params)
#         self.amaf = []
