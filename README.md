# Improving Training-free Neural Architecture Search using Monte-Carlo Methods and Neural Tangent Kernel

This repository is the official implementation of [Improving Training-free Neural Architecture Search using Monte-Carlo Methods and Neural Tangent Kernel]().
We provide several files:

- `main.py` conducts neural architecture search using MONET.
- `experiments.ipynb` replicates the experiments from the paper.
- `search_space_distribution.py` generates a distribution of the search spaces.

## Requirements

- [NAS-Bench-101](https://github.com/vilhess/nasbenchv2) (Edited for new versions of TensorFlow)
- [NAS-Bench-201](https://github.com/D-X-Y/NAS-Bench-201)
- [NATS-Bench](https://github.com/D-X-Y/NATS-Bench)
- [NAS-Bench-301](https://github.com/automl/nasbench301)
- [AutoDL](https://github.com/D-X-Y/AutoDL-Projects)
- [NASBench-PyTorch](https://github.com/romulus0914/NASBench-PyTorch)

## Preparing Data and Training-free Metrics

### CIFAR 10
1. Download the [`cifar10` dataset](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz) and put it on a root folder `Dataset`.

### For NAS-Bench-101
1. Download the [`dataset` directory](https://storage.googleapis.com/nasbench/nasbench_full.tfrecord) and put it into the `API` directory and also clone the [NAS-Bench-101 repo] edited for new tensorflows versions: (https://github.com/vilhess/nasbenchv2) and install the package.
   
### For NAS-Bench-201
1. Download the [`NAS-Bench-201` dataset](https://drive.google.com/open?id=1SKW0Cu0u8-gb18zDpaAGi0f74UdXeGKs) and put in the `API` directory in the root folder of this project.

### For NATS-Bench
1. Download the [`NATS-Bench` dataset](https://drive.google.com/file/d/1IabIvzWeDdDAWICBzFtTCMXxYWPIOIOX/view) and put in the `API` directory in the root folder of this project.

### For NAS-Bench 301
1. Download the [Surrogate Models](https://figshare.com/articles/software/nasbench301_models_v1_0_zip/13061510?file=24992018) and put the directory `nb_models` in the `API`directory


## Usage

The notebook `experiments.ipynb` can be run to replicate the experiments from the paper.
The `main.py` script is designed to accept command-line arguments that configure different aspects of the experimental setup.

### Command-line Arguments

The script accepts the following command-line arguments:

- `--search_space`: Optional. Defines the search space to be used. Must be one of `["nb101", "nb201", "nb301", "nats"]`. The default value is `"nats"`.

- `--algorithm`: Optional. Defines the algorithm to be used. Must be one of `["nrpa", "uct"]`. The default value is `"nrpa"`.

- `--log_dir`: Optional. Specifies the directory where logs will be saved. This should be a string representing the path to the directory. The default value is `"logs"`.

### Output

The script will output the run results to the `log_dir` directory.

