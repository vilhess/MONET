import sys
sys.path.append('/Users/samy/Desktop/NTK4NAS/')

from search_algorithms.RegularizedEvolution import RegularizedEvolution
from search_spaces.nas_bench_301.NASBench301Node import DARTSNode, DARTSCell


import os
import random
from collections import namedtuple

import nasbench301 as nb

class NB301CellRegEvo(DARTSNode):
    def __init__(self, cell1=DARTSCell(), cell2=DARTSCell(), model=None):
        super(NB301CellRegEvo, self).__init__((cell1, cell2))
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

    def mutate(self):
        normal_cell_mutated = self.state[0].mutate_cell()
        reduction_cell_mutated = self.state[1].mutate_cell()
        mutated = NB301CellRegEvo(normal_cell_mutated, reduction_cell_mutated, model=self.performance_model)
        while not mutated.is_terminal():
            av_actions = mutated.get_action_tuples()
            action = random.choice(av_actions)
            mutated.play_action(action)
        return mutated


class RegularizedEvolutionNB301(RegularizedEvolution):
    def __init__(self, performance_model=None, population_size=50, cycles=200, sample_size=25):
        super(RegularizedEvolutionNB301, self).__init__(population_size, cycles, sample_size)
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
        cell = NB301CellRegEvo(model = self.performance_model)
        while not cell.is_complete():
            av_actions = cell.get_action_tuples()
            ac = random.choice(av_actions)
            cell.play_action(ac)
        cell.get_reward()
        return cell

    def mutate(self, model):
        mutated = model.mutate()
        mutated.get_reward()
        return mutated

if __name__=="__main__":

    evolution = RegularizedEvolutionNB301()
    evolution.run()
    evolution.plot()

