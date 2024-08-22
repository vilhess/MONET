from MCTS.Node import Node, AMAFNode, NestedNode
from search_spaces.nas_bench_201.NASBench201Node import NASBench201Node, NASBench201Cell, NASBench201Vertice
import numpy as np
from copy import deepcopy
import random

class NATSBenchTopologyNode(NASBench201Node):
    """
    NATSBenchNode for searching architecture topology. This is identical to NASBench201.
    """

    def __init__(self, state: NASBench201Cell, move=None, parent=None, zobrist_table=None):
        super().__init__(state, move, parent, zobrist_table)


class NATSBenchSizeNode(Node):

    def __init__(self, state: dict = None, move=None, parent=None, zobrist_table=None):
        self.sizes = [8, 16, 24, 32, 40, 48, 56, 64]
        if state is None:
            state = {i: None for i in range(5)}
        super().__init__(state, move, parent)
        self.ADJACENCY_MATRIX_SIZE = 5
        self.N_OPERATIONS = len(self.sizes)
        self.hash = None
        if zobrist_table is not None:
            self.hash = self.calculate_zobrist_hash(zobrist_table)

    def adjacency_matrix(self):
        adj = np.array(list(self.state.values()))
        adj_indexes = []
        for a in adj:
            if a is None:
                adj_indexes.append(0)
            else:
                adj_indexes.append(self.sizes.index(a))
        return adj_indexes

    def add_zobrist(self, zobrist_table):
        ### Normalement cette fonction est seulement censée être utilisée pour la racine.
        if self.hash is None:  # Ne pas recalculer si on a déjà calculé
            self.hash = self.calculate_zobrist_hash(zobrist_table)

    def calculate_zobrist_hash(self, zobrist_table):
        assert zobrist_table is not None, "Remember to pass zobrist_table to node constructor."
        hash = 0
        adjacency = self.adjacency_matrix()
        for i, row in enumerate(adjacency):
            hash ^= zobrist_table[i][row]
        return hash

    def get_action_tuples(self):
        list_actions = []
        for k, v in self.state.items():
            if v is None:
                for size in self.sizes:
                    list_actions.append((k, size))
        return list_actions

    def to_str(self):
        return "{}:{}:{}:{}:{}".format(*list(self.state.values()))

    def play_action(self, action):
        assert len(action) == 2, "Action length should be 2"
        self.state[action[0]] = action[1]

    def is_terminal(self):
        return None not in self.state.values()

    def mutate(self):
        mutated = deepcopy(self)
        id = random.choice(range(5))
        size = random.choice([s for s in self.sizes if s!=self.state[id]])
        mutated.play_action((id, size))
        return mutated

    def get_neighboors(self):
        neighboors = []
        for id in range(5):
            for size in [s for s in self.sizes if s!=self.state[id]]:
                new_cell = deepcopy(self)
                new_cell.play_action((id, size))
                neighboors.append(new_cell)
        return neighboors

class NATSBenchSizeAMAFNode(NATSBenchSizeNode, AMAFNode):

    def __init__(self, state: dict = None, move=None, parent=None, zobrist_table=None):
        AMAFNode.__init__(self, state, move, parent)
        NATSBenchSizeNode.__init__(self, state, move, parent, zobrist_table)


class NATSBenchSizeNestedNode(NATSBenchSizeNode, NestedNode):

    def __init__(self, state: dict = None, move=None, parent=None, sequence=[], zobrist_table=None):
        NATSBenchSizeNode.__init__(self, state, move, parent, zobrist_table)
        NestedNode.__init__(self, state=self.state, move=move, parent=parent, sequence=sequence)