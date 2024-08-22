import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from nasbench_pytorch.model import Network as NBNetwork
from tqdm import tqdm
import ast

# from dataset.CIFAR import CIFAR10Dataset
# from dataset.utils import subset_classes
# from utils.compute_score import compute_score

########

import numpy as np
import torch
from torch import nn
# Works for NASBench201 API -- Some modifications for Basic Model
from torch._functorch.make_functional import make_functional, make_functional_with_buffers
from torch.func import vmap, jacrev, jacfwd, functional_call, vjp, jvp

from utils.ntk_yaoshu import compute_nuc_norm_ntk_from_scratch

def hooklogdet(K, labels=None):
    s, ld = np.linalg.slogdet(K)
    return ld


class NASWOT:

    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.K = np.zeros((batch_size, batch_size))

    def reset(self):
        self.K = np.zeros((self.batch_size, self.batch_size))

    def score(self, network, inputs):
        def counting_forward_hook(module, inp, out):
            try:
                if not module.visited_backwards:
                    return
                if isinstance(inp, tuple):
                    inp = inp[0]
                inp = inp.view(inp.size(0), -1)
                x = (inp > 0).float()
                K = x @ x.t()
                K2 = (1. - x) @ (1. - x.t())
                self.K = self.K + K.cpu().numpy() + K2.cpu().numpy()
            except:
                pass

        def counting_backward_hook(module, inp, out):
            module.visited_backwards = True

        for name, module in network.named_modules():
            if 'ReLU' in str(type(module)):
                # hooks[name] = module.register_forward_hook(counting_hook)
                module.register_forward_hook(counting_forward_hook)
                module.register_backward_hook(counting_backward_hook)

        #network = network.to(DEVICE)
        x = inputs
        x = x.to(DEVICE)
        x2 = torch.clone(x).to(DEVICE)
        jacobs,  y = get_batch_jacobian(network, x)

        network(x)
        s = hooklogdet(self.K)
        return s/100


def get_batch_jacobian(net, x):
    net.zero_grad()
    x.requires_grad_(True)
    y = net(x)[1]
    y.backward(torch.ones_like(y))
    jacob = x.grad.detach()
    return jacob, y.detach()


class Scalar_NN(nn.Module):
    def __init__(self, network, class_val=2, benchmark="nb"):
        super(Scalar_NN, self).__init__()
        self.network = network
        self.class_val = class_val
        self.benchmark = benchmark

    def forward(self, x):
        if self.benchmark == "nb":
            return self.network(x)[-1][:, self.class_val]
        else:
            return self.network(x)[:, self.class_val].reshape(-1, 1)


def model_min_eigenvalue_class2(model, x, class_val):
    net = Scalar_NN(model, class_val, benchmark=None)
    params = {k: v.detach() for k, v in net.named_parameters()}

    for module in net.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.track_running_stats = False

    def fnet_single(params, x):
        return functional_call(net, params, (x.unsqueeze(0),)).squeeze(0)

    jac1 = vmap(jacrev(fnet_single), (None, 0))(params, x)

    jac1 = jac1.values()


    jac1 = [j.flatten(2) if len(j.shape)>2 else j.unsqueeze(2) for j in jac1]

    result = torch.stack([torch.einsum('Naf,Mbf->NMab', j1, j2) for j1, j2 in zip(jac1, jac1)])

    result = result.sum(0)
    u, sigma, v = torch.linalg.svd(result)

    return torch.min(sigma)


def compute_score2(model, dataset_classes, class_permutation, device="cpu"):
    model = model.to(device)
    lambdas = []
    for c in dataset_classes.keys():
        x_ntks = dataset_classes[c]
        if class_permutation is not None:
            c = class_permutation[c]
        lam = model_min_eigenvalue_class2(model, x_ntks, c)
        lambdas.append(lam.cpu().numpy())
    return np.mean(lambdas), np.sum(lambdas)

########

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

if __name__=="__main__":

    DEVICE="cuda:3"

    df = pd.read_csv('nas_bench_101.csv')
    df['score_ntk_mean']=0.0
    df['score_ntk_sum'] = 0.0
    df["score_nasi"]=0.0
    df["score_naswot"]=0.0

    dataset = CIFAR10Dataset()

    criterion = nn.CrossEntropyLoss()

    dataset_classes, class_permutation = subset_classes(dataset, samples_per_class=20, device=DEVICE, subsample=10)

    x_ntk = torch.stack([dataset[i][0].to(DEVICE) for i in range(20)])
    y_ntk = torch.tensor([dataset[i][1].item() if isinstance(dataset[i][1], torch.Tensor) else dataset[i][1] for i in range(20)])

    size = len(df)
    idxs = np.random.choice(size, 20000)

    for i in tqdm(idxs):

        dic = get_info(df, i)
        adj = dic["adj_matrix"]
        op = dic["operations"]
        network = NBNetwork((adj, op)).to(DEVICE)

        score_ntk_mean, score_ntk_sum = compute_score2(network, dataset_classes, class_permutation, device=DEVICE)
        score_nasi = score_nasi = compute_nuc_norm_ntk_from_scratch(network, x_ntk, y_ntk.to(DEVICE), criterion)
        score_naswot = NASWOT(20).score(network, x_ntk)

        df.at[i, "score_ntk_mean"]=score_ntk_mean
        df.at[i, "score_ntk_sum"] = score_ntk_sum
        df.at[i, "score_nasi"]=score_nasi.item()
        df.at[i, "score_naswot"]=score_naswot.item()

    df.to_csv('nb101-full-v4.csv')