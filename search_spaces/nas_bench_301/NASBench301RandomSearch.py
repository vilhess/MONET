import sys
sys.path.append('/Users/samy/Desktop/NTK4NAS/')

from search_algorithms.RandomSearch import RandomSearch
from search_spaces.nas_bench_301.NASBench301Node import DARTSNode, DARTSCell

import os
import random
from collections import namedtuple

import nasbench301 as nb

class NB301CellRandomSearch(DARTSNode):
    def __init__(self, model):
        super(NB301CellRandomSearch, self).__init__((DARTSCell(), DARTSCell()))




class RandomSearchNB301(RandomSearch):
    def __init__(self, performance_model=None, max_iter=100):
        super(RandomSearchNB301, self).__init__(max_iter)
        if performance_model is None: 
            models_1_0_dir = "API/nb_models"
            model_paths = {
                model_name: os.path.join(models_1_0_dir, '{}_v1.0'.format(model_name))
                for model_name in ['xgb', 'lgb_runtime']
            }
            self.performance_model = nb.load_ensemble(model_paths['xgb'])
        else:
            self.performance_model = performance_model

    def get_random_cell(self):
        node = NB301CellRandomSearch(self.performance_model)
        while not node.is_complete():
            av_actions = node.get_action_tuples()
            ac = random.choice(av_actions)
            node.play_action(ac)
        acc, _ = self.get_reward(node)
        return node, acc, _

    def get_reward(self, cell):
        normal_cell_genotype = cell.state[0].to_genotype()
        reduction_cell_genotype = cell.state[1].to_genotype()  # REMPLACER AVEC 1, C'EST JUSTE POUR LE CALCUL DU NTK
        Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')
        genotype_config = Genotype(
                normal=normal_cell_genotype,
                normal_concat=[2, 3, 4, 5],
                reduce=reduction_cell_genotype,
                reduce_concat=[2, 3, 4, 5]
        )
        accuracy_prediction_genotype = self.performance_model.predict(config=genotype_config, representation="genotype",
                                                        with_noise=True)

        acc = accuracy_prediction_genotype
        return acc, None

        self.acc = accuracy_prediction_genotype


# class RandomSearchNB301(RandomSearch):
#     def __init__(self, performance_model=None, max_iter=100):
#         super(RandomSearchNB301, self).__init__()
#         if performance_model is None: 
#             models_1_0_dir = "/userdata/T0259728/projets/nas_ntk/nas_bench_301/nasbench301_models_v1.0/nb_models"
#             model_paths = {
#                 model_name: os.path.join(models_1_0_dir, '{}_v1.0'.format(model_name))
#                 for model_name in ['xgb', 'lgb_runtime']
#             }
#             self.performance_model = nb.load_ensemble(model_paths['xgb'])
#         else:
#             self.performance_model = performance_model
        
#     def get_random_cell(self):
#         node = NB301CellRandomSearch(self.performance_model)
#         while not node.is_complete():
#             av_actions = node.get_action_tuples()
#             ac = random.choice(av_actions)
#             node.play_action(ac)
#         acc, _ = self.get_reward(node)
#         return node, acc, _

        

if __name__=="__main__":
    evolution = RandomSearchNB301(1000)
    evolution.run()
    evolution.plot()