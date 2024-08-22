import sys
sys.path.append('/Users/samy/Desktop/NTK4NAS/')

import numpy as np

from MCTS.Node import AMAFNode, NestedNode
from search_spaces.nas_bench_201.NASBench201Node import NASBench201Node, NASBench201Cell


class NASBench101Vertice:

    def __init__(self, id):
        self.id = id
        self.label = "none"
        self.edges = {i: 0 for i in range(id)}  # 0 ou 1 : connexion avec les autres vertices
        self.OPERATIONS = ["none", "conv1x1-bn-relu", "conv3x3-bn-relu", "maxpool3x3", "input", "output"]
        self.playable_operations = ["conv1x1-bn-relu", "conv3x3-bn-relu", "maxpool3x3"]  # Les labels qu'on peut assigner

    def get_action_tuples(self):
        list_tuples = []
        if self.label == "none":
            for op in self.playable_operations:
                list_tuples.append(("set_label", op))
        for k, v in self.edges.items():
            if v == 0:
                list_tuples.append(("build_edge", k))
        return list_tuples

    def play_action(self, action_name, action):
        if action_name == "set_label":
            self.label = action
        elif action_name == "build_edge":
            k = action
            self.edges[k] = 1


class NASBench101Cell(NASBench201Cell):

    def __init__(self, n_vertices, vertice_type=NASBench101Vertice):
        super().__init__(n_vertices, vertice_type)

        self.vertices[0].play_action("set_label", "input")
        self.vertices[1].play_action("build_edge", 0)
        self.vertices[n_vertices - 1].play_action("set_label", "output")
        self.vertices[n_vertices - 1].play_action("build_edge", n_vertices-2)

    def adjacency_matrix(self):
        adjacency_matrix = np.zeros((self.n_vertices, self.n_vertices), dtype="int8")
        for i, vertice in enumerate(self.vertices):
            for j, connexion in vertice.edges.items():
                if connexion == 1:
                    adjacency_matrix[j, i] = 1
        return adjacency_matrix

    def operations_and_adjacency(self):
        adjacency = self.adjacency_matrix()
        operations = []
        for v in self.vertices:
            operations.append(v.label)

        return adjacency, operations

    def play_action(self, vertice, id, operation):

        self.vertices[vertice].play_action(id, operation)
        super().play_action(vertice, id,  operation)

    def get_action_tuples(self):
        sum_edges = 0
        for v in self.vertices:
            n_edges = int(np.sum(list(v.edges.values())))
            sum_edges += n_edges
        list_tuples = []
        for i, v in enumerate(self.vertices):
            actions = v.get_action_tuples()
            if sum_edges >= 9:
                actions_dup = []
                for act in actions:
                    if act[0] == "set_label":
                        actions_dup.append(act)
                actions = actions_dup
            for action in actions:
                list_tuples.append((i, *action))
        return list_tuples

    def is_complete(self):
        is_complete = True
        sum_edges = 0
        for v in self.vertices:
            n_edges = int(np.sum(list(v.edges.values())))
            sum_edges += n_edges
        if sum_edges > 9:
            is_complete = False
        for v in self.vertices:
            if v.label == "none":
                is_complete = False
        return is_complete


class NASBench101Node(NASBench201Node):

    def __init__(self, state: NASBench101Cell, move=None, parent=None, zobrist_table=None):
        super().__init__(state, move, parent, zobrist_table)
        self.N_OPERATIONS = 5

    def calculate_zobrist_hash(self, zobrist_table):
        hash = 0
        adjacency = self.state.adjacency_matrix()
        for i, element in enumerate(adjacency.flatten()):
            hash ^= zobrist_table[i][element]
        for i, v in enumerate(self.state.vertices):
            op_index = v.OPERATIONS.index(v.label)
            hash ^= zobrist_table[adjacency.shape[0]**2+i][op_index]
        return hash

class NASBench101AMAFNode(NASBench101Node, AMAFNode):

    def __init__(self, state: NASBench101Cell, move=None, parent=None, zobrist_table=None):
        NASBench201Node.__init__(self, state, move, parent, zobrist_table)
        AMAFNode.__init__(self, state, move, parent)


class NASBench101NestedNode(NASBench101Node, NestedNode):

    def __init__(self, state: NASBench101Cell, move=None, parent=None, sequence=None, zobrist_table=None):
        NASBench101Node.__init__(self, state, move, parent, zobrist_table)
        NestedNode.__init__(self, state, move, parent, sequence)


if __name__ == '__main__':
    cell = NASBench101Cell(4)
    node = NASBench101Node(cell)
    print(node.get_action_tuples())
