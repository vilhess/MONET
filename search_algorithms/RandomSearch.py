import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import random

class RandomSearch:
    def __init__(self, max_iter=1000):
        self.max_iter = max_iter
        self.history = []
        self.accuracies = []
        self.best_accs = [0.0]
        
    def get_random_cell(self):
        pass

    def evolve(self):
        for i in tqdm(range(self.max_iter)):
            model, accuracy, _ = self.get_random_cell()
            self.history.append(model)
            self.accuracies.append(accuracy)
            if accuracy > self.best_accs[-1]:
                self.best_accs.append(accuracy)
            else:
                self.best_accs.append(self.best_accs[-1])

    def get_best_model(self):
        models_dict = dict(zip(self.history, self.accuracies))
        best_model = max(models_dict, key=models_dict.get)
        best_accuracy = max(self.accuracies)
        return best_model, best_accuracy

    def run(self):
        self.evolve()
        best_model, best_accuracy = self.get_best_model()
        print(f"Best accuracy: {best_accuracy}")

    def plot(self):
        plt.plot(self.best_accs[1:])
        plt.title('Best Accuracies during time')
        plt.show()
        plt.close()

class RandomSearch_NTK:
    def __init__(self, max_iter=1000):
        self.max_iter = max_iter
        self.history = []
        self.accuracies = []
        self.ntks = []
        self.best_accs = [0.0]
        self.best_ntks = [0.0]
        
    def get_random_cell(self):
        pass

    def evolve(self):
        for i in tqdm(range(self.max_iter)):
            model, accuracy, ntk = self.get_random_cell()
            self.history.append(model)
            self.accuracies.append(accuracy)
            self.ntks.append(ntk)
            if ntk>self.best_ntks[-1]:
                self.best_accs.append(accuracy)
                self.best_ntks.append(ntk)
            else:
                self.best_accs.append(self.best_accs[-1])
                self.best_ntks.append(self.best_ntks[-1])

    def get_best_model(self):
        models_dict = dict(zip(self.history, self.ntks))
        ntk_acc = dict(zip(self.accuracies, self.ntks))
        best_model = max(models_dict, key=models_dict.get)
        best_accuracy = max(ntk_acc, key=ntk_acc.get)
        return best_model, best_accuracy

    def run(self):
        self.evolve()
        best_model, best_accuracy = self.get_best_model()
        print(f"Best accuracy: {best_accuracy}")

    def plot(self):
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))

        axs[0].plot(self.best_ntks)
        axs[0].set_title('Best NTKS during Evolution')

        axs[1].plot(self.best_accs)
        axs[1].set_title('Best Accuracies during Evolution')

        all_time_best_ntk, all_time_best_acc = self.get_best_ntk_acc()

        axs[0].plot([all_time_best_ntk]*len(self.best_ntks), label="Best Score Ever")
        axs[1].plot([all_time_best_acc]*len(self.best_accs), label="Best Accuracy Ever")

        plt.legend()
        plt.show()
        plt.close()