import sys
sys.path.append('/Users/samy/Desktop/NTK4NAS/')


import random
import os
import nasbench301 as nb
from search_algorithms.LocalSearch import LocalSearch
from search_spaces.nas_bench_301.NASBench301Node import DARTSCell, DARTSNode
from collections import namedtuple

class NB301CellLocalSearch(DARTSNode):
    def __init__(self, cell1=DARTSCell(), cell2=DARTSCell(), model=None):
        super(NB301CellLocalSearch, self).__init__((cell1, cell2))
        self.performance_model = model

    def get_reward(self):
        normal_cell_genotype = self.state[0].to_genotype()
        reduction_cell_genotype = self.state[1].to_genotype()  # REMPLACER AVEC 1, C'EST JUSTE POUR LE CALCUL DU NTK
        Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')
        genotype_config = Genotype(
                normal=normal_cell_genotype,
                normal_concat=[2, 3, 4, 5],
                reduce=reduction_cell_genotype,
                reduce_concat=[2, 3, 4, 5]
        )
        accuracy_prediction_genotype = self.performance_model.predict(config=genotype_config, representation="genotype",
                                                        with_noise=True)

        self.acc = accuracy_prediction_genotype

    def get_neighboors(self, model):
        neigh_cell1 = self.state[0].get_cell_neighboors()
        neigh_cell2 = self.state[1].get_cell_neighboors()
        neighs = []
        for c1 in neigh_cell1:
            for c2 in neigh_cell2:
                neighs.append(NB301CellLocalSearch(cell1=c1, cell2=c2, model=model))
        return neighs


class LocalSearchNB301(LocalSearch):
    def __init__(self):
        super(LocalSearchNB301, self).__init__()

        models_1_0_dir = "API/nb_models"
        model_paths = {
            model_name: os.path.join(models_1_0_dir, '{}_v1.0'.format(model_name))
            for model_name in ['xgb', 'lgb_runtime']
        }
        self.performance_model = nb.load_ensemble(model_paths['xgb'])

    def get_random_cell(self):
        cell = NB301CellLocalSearch(model=self.performance_model)
        while not cell.is_complete():
            av_actions = cell.get_action_tuples()
            ac = random.choice(av_actions)
            cell.play_action(ac)
        cell.get_reward()
        return cell
        
    def get_neighboors(self, model):
        neighboors = model.get_neighboors(self.performance_model)
        for neig in neighboors:
            neig.get_reward()
        return neighboors


if __name__=="__main__":
    search = LocalSearchNB301()
    search.run()
    search.plot()