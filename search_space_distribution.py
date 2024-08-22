import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from nasbench_pytorch.model import Network as NBNetwork
from xautodl.models import get_cell_based_tiny_net
from nas_201_api import NASBench201API as API
from nats_bench import create
# import nasbench301 as nb
# from DARTS.search_cnn import get_random_node, get_genotype, NetworkCIFAR
from tqdm import tqdm
import ast
import os
import itertools
from collections import namedtuple

from utils.CIFAR import CIFAR10Dataset
from utils.helpers import subset_classes, configure_seaborn, normalize

from ntk.compute_score import compute_score

configure_seaborn()


def get_info(df, index):
    row = df.iloc[index]
    adj = ast.literal_eval(row["module_adjacency"].replace(' ', ',').replace('\n', ''))
    op = eval(row["module_operations"])
    num_params = row["trainable_parameters"]
    valid_acc = row["final_validation_accuracy"]
    test_acc = row["final_test_accuracy"]
    return {
        "adj_matrix":adj,
        "operations":op,
        "num_params":num_params,
        "valid_acc":valid_acc,
        "test_acc":test_acc
    }

# def get_accuracy(node, model):
#     normal_cell_genotype = node.state[0].to_genotype()
#     reduction_cell_genotype = node.state[1].to_genotype()
#     Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')
#     genotype_config = Genotype(
#         normal=normal_cell_genotype,
#         normal_concat=[2, 3, 4, 5],
#         reduce=reduction_cell_genotype,
#         reduce_concat=[2, 3, 4, 5]
#     )
#     accuracy_prediction_genotype = model.predict(config=genotype_config, representation="genotype",
#                                                         with_noise=True)
#     return accuracy_prediction_genotype


if __name__=="__main__":

    if not os.path.isdir("logs_dir"):
        os.mkdir("log_dir")

    DEVICE=torch.device("cpu")

    # models_1_0_dir = "../../Downloads/nb_models"
    # model_paths = {
    #     model_name: os.path.join(models_1_0_dir, '{}_v1.0'.format(model_name))
    #     for model_name in ['xgb', 'lgb_runtime']
    # }
    # performance_model = nb.load_ensemble(model_paths['xgb'])

    dataset = CIFAR10Dataset(data_dir="../Dataset/CIFAR10/cifar-10-batches-py")

    dataset_classes, class_permutation = subset_classes(dataset, samples_per_class=2, device=DEVICE, subsample=10)

    print("Currently working on NASBench 101")

    # NASBENCH 101

    df = pd.read_csv('API/nas_bench_101.csv')

    df['score_ntk'] = 0.0

    size = len(df)
    idxs = range(size)

    for i in tqdm(idxs):

        dic = get_info(df, i)
        adj = dic["adj_matrix"]
        op = dic["operations"]
        network = NBNetwork((adj, op)).to(DEVICE)

        score_ntk = compute_score(network, dataset_classes, class_permutation, device=DEVICE, benchmark="nb101")

        df.at[i, "score_ntk"] = score_ntk

        nb101 = df.iloc[idxs]


    # NASBENCH 201

    print("Currently working on NASBench 201")

    nb201 = {        "index":[],
                    "score_ntk":[],
                    "accuracy":[]}

    api = API('API/NAS-Bench-201-v1_1-096897.pth', verbose=False)

    size = len(api)
    idxs = range(size)

    for idx in tqdm(idxs):

        config = api.get_net_config(idx, 'cifar10')
        network = get_cell_based_tiny_net(config).to(DEVICE)

        info = api.query_meta_info_by_index(idx, hp="200")
        accuracy = info.get_metrics("cifar10-valid", 'valid')['accuracy']/100

        nb201['index'].append(idx)
        nb201['accuracy'].append(accuracy)

        score_ntk = compute_score(network, dataset_classes, class_permutation, device=DEVICE, benchmark="nb201")
        nb201['score_ntk'].append(score_ntk)

    # NATS

    print("Currently working on NATS")

    nats = {        "index":[],
                    "score_ntk":[],
                    "accuracy":[]}
    
    api = create("API/NATS-sss-v1_0-50262.pickle.pbz2", 'sss', fast_mode=False, verbose=False)

    idxs = range(len(api))

    for idx in tqdm(idxs):

        config = api.get_net_config(idx, 'cifar10')
        network = get_cell_based_tiny_net(config).to(DEVICE)

        info = api.get_more_info(idx, 'cifar10', hp="90")
        accuracy = info['test-accuracy']/100

        nats['index'].append(idx)
        nats['accuracy'].append(accuracy)

        score_ntk = compute_score(network, dataset_classes, class_permutation, device=DEVICE, benchmark="nats")
        nats["score_ntk"].append(score_ntk)

    
    # # NASBench 301

    # nb301 = {
    #     "normal_cell":[],
    #     "reduction_cell":[],
    #     "num_params":[],
    #     "accuracy":[],
    #     "score_ntk":[],
    # }
    
    # for _ in tqdm(range(3)):
    #     node = get_random_node()
    #     genotype = get_genotype(node)

    #     list_act_normal = [list(v.actions.values()) for v in node.state[0].vertices[2:]]
    #     list_act_normal = list(itertools.chain.from_iterable(list_act_normal))

    #     list_act_reduce = [list(v.actions.values()) for v in node.state[1].vertices[2:]]
    #     list_act_reduce = list(itertools.chain.from_iterable(list_act_reduce))

    #     network = network = NetworkCIFAR(3, 10, 5, False, genotype).to(DEVICE)
    #     num_params = sum(p.numel() for p in network.parameters())
        
    #     acc = get_accuracy(node, performance_model)

    #     score_ntk = compute_score(network, dataset_classes, class_permutation, device=DEVICE, benchmark="darts")

    #     nb301["normal_cell"].append(list_act_normal)
    #     nb301["reduction_cell"].append(list_act_reduce)
    #     nb301["num_params"].append(num_params)
    #     nb301["accuracy"].append(acc)
    #     nb301["score_ntk"].append(score_ntk.item())

    # Plot 

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    axs[2].scatter(normalize(nb101['score_ntk']), nb101['final_validation_accuracy'], color=sns.color_palette()[0])
    axs[2].set_title('NASBench 101')
    axs[2].legend()

    axs[0].scatter(normalize(nb201['score_ntk']), nb201['accuracy'], color=sns.color_palette()[0])
    axs[0].set_title('NASBench 201')
    axs[0].legend()

    axs[1].scatter(normalize(nats['score_ntk']), nats['accuracy'], color=sns.color_palette()[0])
    axs[1].set_title('NATS Bench')
    axs[1].legend()

    # axs[3].scatter(normalize(nb301['score_ntk']), nb301['accuracy'], color=sns.color_palette()[0])
    # axs[3].set_title('NASBench 301')
    # axs[3].legend()

    plt.legend()
    plt.savefig(os.path.join("log_dir", "search_space_distribution.png"))