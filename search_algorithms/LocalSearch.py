import matplotlib.pyplot as plt

class LocalSearch:
    def __init__(self, num_init=10):
        self.best_accs_history = [0.0]
        self.best_accs = [0.0]
        self.num_init = num_init
        
    def get_random_cell(self):
        pass

    def get_neighboors(self, model):
        pass

    def get_best_acc(self):
        pass

    def local_search(self):

        best_candidate = None
        init_best_score = 0
        for _ in range(self.num_init):
            cell = self.get_random_cell()
            if cell.acc > init_best_score:
                best_candidate = cell 
                init_best_score = cell.acc
            self.best_accs_history.append(init_best_score)
        best_score = best_candidate.acc
        self.best_accs.append(best_candidate.acc)
        while self.best_accs[-1]>self.best_accs[-2]:
            candidates = self.get_neighboors(best_candidate)
            for candidate in candidates:
                score = candidate.acc
                if score>best_score:
                    best_score=score
                    best_candidate=candidate
                self.best_accs_history.append(best_score)
            self.best_accs.append(best_score)
        return best_candidate

    def run(self):
        best_candidate = self.local_search()
        print(f"Best accuracy: {best_candidate.acc}")

    def plot(self):
        plt.plot(self.best_accs_history[1:])
        plt.title('Best Accuracies during time')
        plt.legend()
        plt.show()



class LocalSearch_NTK:
    def __init__(self, num_init=10):
        self.best_accs_history = [0.0]
        self.best_ntks_history = [0.0]
        self.best_ntks = [0.0]
        self.num_init = num_init
        
    def get_random_cell(self):
        pass

    def get_neighboors(self, model):
        pass

    def get_best_acc(self):
        pass

    def local_search(self):

        best_candidate = None
        init_best_score = 0
        init_best_acc = 0

        for _ in range(self.num_init):
            cell = self.get_random_cell()
            if cell.ntk > init_best_score:
                best_candidate = cell 
                init_best_score = best_candidate.ntk 
                init_best_acc = best_candidate.acc 
            self.best_accs_history.append(init_best_acc)
            self.best_ntks_history.append(init_best_score)
        best_score = best_candidate.ntk
        best_acc = best_candidate.acc
        self.best_ntks.append(best_score)
        while self.best_ntks[-1]>self.best_ntks[-2]:
            candidates = self.get_neighboors(best_candidate)
            for candidate in candidates:
                score = candidate.ntk
                if score>best_score:
                    best_score=score
                    best_acc = candidate.acc
                    best_candidate=candidate
                self.best_ntks_history.append(best_score)
                self.best_accs_history.append(best_acc)
            self.best_ntks.append(best_score)
        return best_candidate

    def run(self):
        best_candidate = self.local_search()
        print(f"Best accuracy: {best_candidate.acc}")

    def plot(self):
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))

        axs[0].plot(self.best_ntks_history[1:])
        axs[0].set_title('Best NTKS during Evolution')

        axs[1].plot(self.best_accs_history[1:])
        axs[1].set_title('Best Accuracies during Evolution')

        plt.legend()
        plt.show()
        plt.close()