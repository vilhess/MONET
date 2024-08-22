import numpy as np
from graphviz import Digraph

from MCTS.Node import Node, AMAFNode, NestedNode
from itertools import chain
import random
from copy import deepcopy



class NASBench201Vertice:

    def __init__(self, id):
        self.id = id
        self.actions = {i: None for i in range(id)}
        self.OPERATIONS = ["none", "skip_connect", "nor_conv_1x1", "nor_conv_3x3", "avg_pool_3x3"]


    def is_complete(self):
        return None not in self.actions.values()

    def play_action(self, id, operation):
        self.actions[id] = operation

    def get_action_tuples(self):
        list_tuples = []
        for k, v in self.actions.items():
            if v is None:
                for op in self.OPERATIONS:
                    list_tuples.append((k, op))
        return list_tuples


class NASBench201Cell:

    def __init__(self, n_vertices=4, vertice_type=NASBench201Vertice):
        self.n_vertices = n_vertices
        self.vertices = [vertice_type(i) for i in range(n_vertices)]
        self.OPERATIONS = self.vertices[0].OPERATIONS

    def is_complete(self):
        return all([v.is_complete() for v in self.vertices])

    def to_str(self):
        assert self.is_complete(), "cell is incomplete"
        res = ""
        for v in self.vertices[1:]:
            temp_str = "|"
            # sorted_actions = collections.OrderedDict(sorted(d.items()))
            for source, operation in v.actions.items():
                temp_str += f"{operation}~{source}|"
            res += temp_str + "+"
        return res[:-1]  # -1 pour enlever le dernier "+"

    def adjacency_matrix(self):
        adjacency_matrix = np.zeros((self.n_vertices, self.n_vertices), dtype="int8")
        for i, vertice in enumerate(self.vertices):
            for j, operation in vertice.actions.items():
                if operation is None:
                    operation = "none"
                adjacency_matrix[j,i] = self.OPERATIONS.index(operation)
        return adjacency_matrix

    def play_action(self, vertice, id, operation):
        """
        vertice : the vertice that plays the action
        id: the vertice that acts as input for vertice
        """
        self.vertices[vertice].play_action(id, operation)

    def get_action_tuples(self):
        list_tuples = []
        for i, v in enumerate(self.vertices):
            actions = v.get_action_tuples()
            for action in actions:
                list_tuples.append((i, *action))
        return list_tuples

    def mutate(self):
        mutated = deepcopy(self)
        vertice = random.choice(range(1, mutated.n_vertices))
        id = random.choice(range(vertice))
        action = random.choice([op for op in mutated.OPERATIONS if op!=self.vertices[vertice].actions[id]])
        mutated.play_action(vertice, id, action)
        return mutated

    def get_neighboors(self):
        neighboors = []
        for vertice in range(1, self.n_vertices):
            for id in range(vertice):
                for action in [op for op in self.OPERATIONS if op!=self.vertices[vertice].actions[id]]:
                    new_cell = deepcopy(self)
                    new_cell.play_action(vertice, id, action)
                    neighboors.append(new_cell)
        return neighboors


    def plot(self, filename="cell"):
        g = Digraph(
                format='pdf',
                edge_attr=dict(fontsize='20', fontname="garamond"),
                node_attr=dict(style='rounded, filled', shape='rect', align='center', fontsize='20', height='0.5',
                               width='0.5', penwidth='2', fontname="garamond"),
                engine='dot')
        g.body.extend(['rankdir=LR'])

        g.node("c_{k-1}", fillcolor='darkseagreen2')
        g.node("c_{k}", fillcolor='palegoldenrod')

        steps = self.n_vertices - 2

        for i in range(steps):
            g.node(str(i + 1), fillcolor='lightblue')

        for i, vertice in enumerate(self.vertices):
            for k, v in vertice.actions.items():
                # print(str(i), str(k), v)
                in_ = str(k)
                out_ = str(i)
                if k == 0:
                    in_ = "c_{k-1}"
                if i == self.n_vertices - 1:
                    out_ = "c_{k}"
                g.edge(in_, out_, label=v, fillcolor="gray")

        g.render(filename, view=True)

    def get_reward(self, api, df):
        pass

class NASBench201Node(Node):

    def __init__(self, state: NASBench201Cell, move=None, parent=None, zobrist_table=None):
        super().__init__(state, move, parent)
        N_NODES = state.n_vertices
        self.ADJACENCY_MATRIX_SIZE = N_NODES**2
        self.N_OPERATIONS = 5
        self.hash = None
        if zobrist_table is not None:
            self.hash = self.calculate_zobrist_hash(zobrist_table)

    def add_zobrist(self, zobrist_table):
        ### Normalement cette fonction est seulement censée être utilisée pour la racine.
        if self.hash is None:  # Ne pas recalculer si on a déjà calculé
            self.hash = self.calculate_zobrist_hash(zobrist_table)

    def calculate_zobrist_hash(self, zobrist_table):
        assert zobrist_table is not None, "Remember to pass zobrist_table to node constructor."
        hash = 0
        adjacency = self.state.adjacency_matrix()
        for i, row in enumerate(adjacency):
            for element in row:
                hash ^= zobrist_table[i][element]
        return hash

    def get_action_tuples(self):
        return self.state.get_action_tuples()

    def play_action(self, action):
        assert len(action) == 3, "Action length should be 3."
        self.state.play_action(*action)

class NASBench201AMAFNode(NASBench201Node, AMAFNode):

    def __init__(self, state: NASBench201Cell, move=None, parent=None, zobrist_table=None):
        NASBench201Node.__init__(self, state, move, parent, zobrist_table)
        AMAFNode.__init__(self, state, move, parent)


class NASBench201NestedNode(NASBench201Node, NestedNode):

    def __init__(self, state: NASBench201Cell, move=None, parent=None, sequence=None, zobrist_table=None):
        NASBench201Node.__init__(self, state, move, parent, zobrist_table)
        NestedNode.__init__(self, state, move, parent, sequence)

if __name__ == '__main__':
    root_node = NASBench201NestedNode(state=NASBench201Cell(4))
    print(root_node)
