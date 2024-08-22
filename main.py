import argparse

from nasbench import api  # NB101
from nas_201_api import NASBench201API as API  # NB201
import nasbench301 as nb  # NB301
import nats_bench  # NATS

# NAS-Bench-101
from search_spaces.nas_bench_101.NASBench101Node import *
from search_spaces.nas_bench_101.NASBench101Node import NASBench101Node as nb101node

from search_spaces.nas_bench_101.NASBench101MCTS import *
from search_spaces.nas_bench_101.NASBench101RandomSearch import *
from search_spaces.nas_bench_101.NASBench101RegEvo import *

# NAS-Bench-201
from search_spaces.nas_bench_201.NASBench201Node import *
from search_spaces.nas_bench_201.NASBench201MCTS import *
from search_spaces.nas_bench_201.NASBench201RandomSearch import *
from search_spaces.nas_bench_201.NASBench201RegEvo import *

# NAS-Bench-301
from search_spaces.nas_bench_301.NASBench301Node import *
from search_spaces.nas_bench_301.NASBench301MCTS import *
from search_spaces.nas_bench_301.NASBench301RandomSearch import *
from search_spaces.nas_bench_301.NASBench301RegEvo import *

# NATS-Bench
from search_spaces.nats_bench_dataset.NATSBenchNode import *
from search_spaces.nats_bench_dataset.NATSBenchMCTS import *
from search_spaces.nats_bench_dataset.NATSBenchRandomSearch import *
from search_spaces.nats_bench_dataset.NATSBenchRegEvo import *

from MCTS.mcts_agent import *
from MCTS.nested import *

from utils.helpers import normalize


def parse_arguments():
    parser = argparse.ArgumentParser(description="Process command-line arguments for the project.")

    # Adding the "search_space" argument
    parser.add_argument(
        "--search_space",
        type=str,
        choices=["nb101", "nb201", "nb301", "nats"],
        default="nats",
        help="The search space to use. Must be one of ['nb101', 'nb201', 'nb301', 'nats']. Default is 'nats'."
    )

    # Adding the "algorithm" argument
    parser.add_argument(
        "--algorithm",
        type=str,
        choices=["nrpa", "uct", "regevo", "random"],
        default="nrpa",
        help="The algorithm to use. Must be one of ['nrpa', 'uct', 'regevo', 'random']. Default is 'nrpa'."
    )

    # Adding the "log_dir" argument
    parser.add_argument(
        "--log_dir",
        type=str,
        default="logs",
        help="The directory where logs will be saved. Default is 'logs'."
    )

    # Adding the "params" argument
    parser.add_argument(
        "--params",
        type=str,
        default="utils/params.json",
        help="The directory where params are run. Default is 'utils/params.json'."
    )

    args = parser.parse_args()

    # Convert the arguments to a dictionary
    args_dict = vars(args)

    return args_dict, parser


matching_dict_sspace = {"nb101": "NASBench101",
                        "nb201": "NASBench201",
                        "nb301": "NASBench301",
                        "nats": "NATSBench"}

matching_dict_algo = {"nrpa": "NRPA",
                      "uct": "UCT",
                      "regevo": "RegEvo",
                      "random": "RandomSearch"}

if __name__ == "__main__":

    args_dict, parser = parse_arguments()
    params = args_dict["params"]

    if not os.path.isdir(args_dict["log_dir"]):
        os.mkdir(args_dict["log_dir"])

    if args_dict["search_space"] == "nb201":
        api = API('API/NAS-Bench-201-v1_1-096897.pth', verbose=False)


        if args_dict["algorithm"] == "nrpa":
            root_node = NASBench201NestedNode(NASBench201Cell() , sequence=[])
            agent = NASBench201NRPA_NTK(root_node, level=3, api=api, params_path=params)
            agent.n_iter = int(np.ceil(np.power(agent.n_iter, 1/3)))
        
        if args_dict["algorithm"] == "uct":
            root_node = NASBench201Node(NASBench201Cell())
            agent = NASBench201UCT_NTK(root_node, api=api, params_path=params)
            agent.n_iter = 20

        # if args_dict["algorithm"] == "regevo":
        #     agent = RegularizedEvolutionNB201_NTK(api)

    elif args_dict["search_space"] == "nb301":
        models_1_0_dir = "API/nasbench301_models_v1.0/nb_models/"
        model_paths = {
            model_name: os.path.join(models_1_0_dir, '{}_v1.0'.format(model_name))
            for model_name in ['xgb', 'lgb_runtime']
        }
        NB_301_performance_model = nb.load_ensemble(model_paths['xgb'])

        if args_dict["algorithm"] == "nrpa":
            root_node = DARTSNestedNode((DARTSCell(), DARTSCell()), sequence=[])
            agent = NASBench301NRPA_NTK(root_node, level=3, params_path=params)
            agent.n_iter = int(np.ceil(np.power(agent.n_iter, 1/3)))

        elif args_dict["algorithm"] == "uct":
            root_node = DARTSNode((DARTSCell(), DARTSCell()))
            agent = NASBench301UCT_NTK(root_node, performance_model=NB_301_performance_model, save_folder=None, params_path=params, disable_tqdm=False)
            agent.n_iter = 20

    elif args_dict["search_space"] == "nats":
        nats_bench_api = nats_bench.create("API/NATS-sss-v1_0-50262-simple", "sss", fast_mode=True, verbose=False)

        if args_dict["algorithm"] == "nrpa":
            root_node = NATSBenchSizeNestedNode(sequence=[])
            agent = NATSBenchNRPA_NTK(root_node, level=3, api=nats_bench_api, params_path=params)
            agent.n_iter = int(np.ceil(np.power(agent.n_iter, 1/3)))

        elif args_dict["algorithm"] == "uct":
            root_node = NATSBenchSizeNode()
            agent = NATSBenchUCT_NTK(root_node, api=nats_bench_api, params_path=params)

    elif args_dict["search_space"] == "nb101":
        nas_bench_101_api = api.NASBench("API/nasbench_full.tfrecord")

        if args_dict["algorithm"] == "nrpa":
            root_node = NASBench101NestedNode(state=NASBench101Cell(n_vertices=7), sequence=[])
            agent = NASBench101NRPA_NTK(root_node, level=3, api=nas_bench_101_api, params_path=params)
            agent.n_iter = int(np.ceil(np.power(agent.n_iter, 1/3)))


        if args_dict["algorithm"] == "uct":
            root_node = nb101node(state=NASBench101Cell(n_vertices=7))
            agent = NASBench101UCT_NTK(root_node, nas_bench_101_api, save_folder=None, params_path="utils/params.json")

    print(args_dict)
    agent.main_loop()

    results_dict = {"best_reward": [np.float64(e) for e in agent.best_reward], 
                    "best_accuracy": [np.float64(e) for e in agent.best_accuracy]}
                    
    with open(os.path.join(args_dict["log_dir"], "results.json"), "w+") as f:
        json.dump(results_dict, f, indent=4)

    f, ax1 = plt.subplots()
    ax1.plot(normalize(np.array(agent.best_reward)), label="NTK Metric", color="#3d405b")
    ax1.set_ylabel("NTK Metric")

    ax2 = ax1.twinx()
    ax2.plot(np.array(agent.best_accuracy), label="Accuracy", color="#e07a5f")
    ax2.set_ylabel("CIFAR-10 accuracy")

    plt.title("Evolution of the best architecture reward"); plt.xlabel("Iteration"); plt.ylabel("Accuracy"); 
    plt.legend();
    plt.savefig(os.path.join(args_dict["log_dir"], "results.png"))
    # plt.show()