import copy
import os
import random
from datetime import datetime

import numpy as np
import networkx as nx
from decimal import Decimal
import pydot
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import Dataset



def calculate_padding(input_size, output_size, kernel_size, stride, mode="same"):
    if mode == "same":
        return int(np.ceil((stride * (output_size - 1) - input_size + kernel_size) / 2))


def create_now_folder(path):
    folder = datetime.now().strftime('%Y-%m-%d %H-%M-%S')
    os.mkdir(os.path.join(path, folder))
    return folder


def scatterplot_results_vs_amaf(nodes):
    f, ax = plt.subplots(1, 1, figsize=(20,20))
    results = [np.mean(u.results) for u in nodes]
    amaf = [np.mean(u.amaf) for u in nodes]
    names = [u.move[2:] for u in nodes]
    amaf_visits = [len(u.amaf) for u in nodes]
    results_visits = [len(u.results) for u in nodes]
    ax.scatter(x=results, y=amaf)
    for i, txt in enumerate(names):
        ax.annotate(txt, (results[i], amaf[i]))
    ax.set_xlabel("Results")
    ax.set_ylabel("AMAF")
    return f

def scatterplot_results(nodes):
    f, ax = plt.subplots(1, 1, figsize=(20,20))
    results = [np.mean(u.results) for u in nodes]
    names = [u.move[2:] for u in nodes]
    results_visits = [len(u.results) for u in nodes]
    ax.scatter(results, results_visits)
    for i, txt in enumerate(names):
        ax.annotate(txt, (results[i], results_visits[i]))
    ax.set_xlabel("Results")
    ax.set_ylabel("Nombre de visites")
    return f


def plot_mcts_tree(mcts, max_depth=5, filename=None):
    G = nx.DiGraph()
    root = mcts.root
    # G.add_node(root)
    queue = copy.deepcopy(root.children)
    queue = [e for e in queue if len(e.results) != 0]
    while len(queue) != 0:
        v = queue.pop(0)
        # Add v to graph
        G.add_node(v)
        G.add_edge(v.parent, v)

        # Add children of v to the graph
        for c in v.children:
            try:
                if len(c.amaf) != 0 or len(c.results) != 0 or np.mean(c.results) == 0:
                    queue.append(c)
            except Exception:
                if len(c.results) != 0 or np.mean(c.results) == 0:
                    queue.append(c)
    pg = nx.nx_pydot.to_pydot(G)
    for node, nx_node in zip(pg.get_nodes(), G.nodes()):
        try:
            node.set_label(f"Mean result ({len(nx_node.results)}) : {Decimal(np.mean(nx_node.results)):.3E}, AMAF ({len(nx_node.amaf)}) : {Decimal(np.mean(nx_node.amaf)):.3E},")
        except Exception:
            node.set_label(f"Mean result ({len(nx_node.results)}) : {Decimal(np.mean(nx_node.results)):.3E}")
    for edge, nx_edge in zip(pg.get_edges(), G.edges()):
        edge.set_label(str(nx_edge[1].move))
    file = pg.write_png(filename)

def ewma(data, window):

    alpha = 2 /(window + 1.0)
    alpha_rev = 1-alpha
    n = data.shape[0]

    pows = alpha_rev**(np.arange(n+1))

    scale_arr = 1/pows[:-1]
    offset = data[0]*pows[1:]
    pw0 = alpha*alpha_rev**(n-1)

    mult = data*pw0*scale_arr
    cumsums = mult.cumsum()
    out = offset + cumsums*scale_arr[::-1]
    return out

def running_avg(data, window_size):
    res = np.zeros(len(data)-window_size)
    sum = 0
    for i in range(window_size):
        sum += data[i]
    for i in range(len(data)-window_size):
        res[i] = sum / window_size
        sum -= data[i]
        sum += data[i + window_size]
    return res

def configure_seaborn(**kwargs):
    sns.set_context("notebook")
    sns.set_theme(sns.plotting_context("notebook", font_scale=1), style="whitegrid")
    palette = ["#3D405B", "#E08042", "#54AB69", "#CE2F49", "#A26EBF", "#7D4948", "#D12AA2"]
    sns.set_palette(palette)


# Return a dictionnary : key -> class / item -> torch batch of samples from this class
def subset_classes(dataset: Dataset, samples_per_class=10, device="cpu", subsample=12):
    dataset_classes = {}
    count_per_class = {}
    class_permutation = None

    for inp, tar in dataset:
        try:
            if tar not in dataset_classes:
                dataset_classes[tar] = []
                count_per_class[tar] = 0
            if count_per_class[tar] < samples_per_class:
                dataset_classes[tar].append(inp.to(device))
                count_per_class[tar] += 1
        except    Exception as e:
            print(f"Error with target {tar} : {e}")

        if all(count >= samples_per_class for count in count_per_class.values()):
            break

    if len(dataset_classes) > subsample:
        selected_classes = random.sample(list(dataset_classes.keys()), subsample)
        dataset_classes = {key: dataset_classes[key] for key in selected_classes}
        class_permutation = {selected_classes[i]: i for i in range(len(selected_classes))}

    for key in dataset_classes.keys():
        dataset_classes[key] = torch.stack(dataset_classes[key])

    return dataset_classes, class_permutation

def z_normalize(data):
    return (data - np.mean(data)) / np.std(data)

def normalize(data, log=False):
    if log is True:
        data = np.log(data)
    return (data - np.min(data)) / (np.max(data) - np.min(data))