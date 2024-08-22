import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import random

class RegularizedEvolution:
    def __init__(self, population_size, cycles, sample_size):
        self.population_size = population_size
        self.cycles = cycles
        self.sample_size = sample_size
        self.population = []
        self.history = []
        self.accuracies = []
        self.best_accs = [0.0]
        
    def get_random_cell(self):
        pass

    def mutate(self, model):
        pass

    def get_best_acc(self):
        pass
    
    def initialize_population(self):
        while len(self.population) < self.population_size:
            random_cell = self.get_random_cell()
            self.population.append(random_cell)
            self.history.append(random_cell)
            if random_cell.acc > self.best_accs[-1]:
                self.best_accs.append(random_cell.acc)
            else:
                self.best_accs.append(self.best_accs[-1])

    def evolve(self):
        while len(self.history) < self.cycles:
            if len(self.history) % 1000 == 0:
                print(f"{len(self.history)}: Best acc : {self.best_accs[-1]}")
            samples = random.sample(self.population, self.sample_size)
            best_parent = max(samples, key=lambda mod: mod.acc)
            if best_parent.acc > self.best_accs[-1]:
                self.best_accs.append(best_parent.acc)
            else:
                self.best_accs.append(self.best_accs[-1])
            mutated = self.mutate(best_parent)
            self.population.append(mutated)
            self.history.append(mutated)
            self.population.pop(0)

    def get_best_model(self):
        best_model = max(self.history, key=lambda model: model.acc)
        return best_model

    def run(self):
        self.initialize_population()
        self.evolve()
        best_model = self.get_best_model()
        print(f"Best accuracy: {best_model.acc}")

    def plot(self):
        plt.plot(self.best_accs[1:])
        plt.title('Best Accuracies during time')
        # all_time_best_acc = self.get_best_acc()
        # plt.plot([all_time_best_acc]*len(self.best_accs), label="Best Accuracy Ever")
        plt.legend()
        plt.show()


class RegularizedEvolutionNTK:
    def __init__(self, population_size, cycles, sample_size):
        self.population_size = population_size
        self.cycles = cycles
        self.sample_size = sample_size
        self.population = []
        self.history = []
        self.best_accs = [0.0]
        self.best_ntks = [0.0]
        
    def get_random_cell(self):
        pass

    def mutate(self, model):
        pass

    def get_best_ntk_acc(self):
        pass
    
    def initialize_population(self):
        while len(self.population) < self.population_size:
            random_cell = self.get_random_cell()
            self.population.append(random_cell)
            self.history.append(random_cell)
            if random_cell.ntk>self.best_ntks[-1]:
                self.best_ntks.append(random_cell.ntk)
                self.best_accs.append(random_cell.acc)
            else:
                self.best_ntks.append(self.best_ntks[-1])
                self.best_accs.append(self.best_accs[-1])

    def evolve(self):
        while len(self.history) < self.cycles:
            if len(self.history) % 1000 == 0:
                print(f"{len(self.history)}: Best acc : {self.best_accs[-1]}")
            samples = random.sample(self.population, self.sample_size)
            best_parent = max(samples, key=lambda mod: mod.ntk)
            mutated = self.mutate(best_parent)
            if mutated.ntk>self.best_ntks[-1]:
                self.best_ntks.append(mutated.ntk)
                self.best_accs.append(mutated.acc)
            else:
                self.best_ntks.append(self.best_ntks[-1])
                self.best_accs.append(self.best_accs[-1])     
            self.population.append(mutated)
            self.history.append(mutated)
            self.population.pop(0)

    def get_best_model(self):
        best_model = max(self.history, key=lambda model: model.acc)
        return best_model

    def run(self):
        self.initialize_population()
        self.evolve()
        best_model = self.get_best_model()
        print(f"Best accuracy: {best_model.acc}")

    def plot(self):
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))

        axs[0].plot(self.best_ntks[1:])
        axs[0].set_title('Best NTKS during Evolution')

        axs[1].plot(self.best_accs[1:])
        axs[1].set_title('Best Accuracies during Evolution')

        # all_time_best_ntk, all_time_best_acc = self.get_best_ntk_acc()

        # axs[0].plot([all_time_best_ntk]*len(self.best_ntks), label="Best Score Ever")
        # axs[1].plot([all_time_best_acc]*len(self.best_accs), label="Best Accuracy Ever")

        plt.legend()
        plt.show()
        plt.close()