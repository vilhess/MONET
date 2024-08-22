import numpy as np
import torch
from torch import nn
# Works for NASBench201 API -- Some modifications for Basic Model
from torch._functorch.make_functional import make_functional, make_functional_with_buffers
from torch.func import vmap, jacrev, jacfwd, functional_call, vjp, jvp


class Scalar_NN(nn.Module):
    def __init__(self, network, class_val, benchmark):
        super(Scalar_NN, self).__init__()
        self.network = network
        self.class_val = class_val
        self.benchmark = benchmark

    def forward(self, x):
        
        if self.benchmark == "nb201" or self.benchmark == "nats":
            return self.network(x)[-1][:, self.class_val]
        else:
            return self.network(x)[:, self.class_val].reshape(-1, 1)

def model_min_eigenvalue_class(model, x, class_val, benchmark):

    model = Scalar_NN(network=model, class_val=class_val, benchmark=benchmark)

    if benchmark!="nb101" and benchmark!="darts":
        def fnet_single(params, x):
            return functional_call(model, params, x.unsqueeze(0))[-1].squeeze(0)
    else:
        def fnet_single(params, x):
            return functional_call(model, params, (x.unsqueeze(0), )).squeeze(0)      

    parameters = {k: v.detach() for k, v in model.named_parameters()}

    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.track_running_stats = False

    jac1 = vmap(jacrev(fnet_single), (None, 0))(parameters, x)
    if benchmark!="nb101" and benchmark!="darts":
        jac1 = tuple(jac1[name] for name in jac1)
        jac1 = [j.flatten(1) for j in jac1]
    elif benchmark=="nb101" or benchmark=="darts":
        jac1 = jac1.values()
        jac1 = [j.flatten(2) if len(j.shape)>2 else j.unsqueeze(2) for j in jac1]

    jac2 = jac1
    if benchmark=="nb101" or benchmark=="darts":
        operation = 'Naf,Mbf->NMab'
    else:
        operation = 'Na,Mb->NM'
    result = torch.stack([torch.einsum(operation, j1, j2) for j1, j2 in zip(jac1, jac2)])
    result = result.sum(0)
    if benchmark=="nb101" or benchmark=="darts":
        result=result.squeeze()
    u, sigma, v = torch.linalg.svd(result)

    return torch.min(sigma)


def compute_score(model, dataset_classes, class_permutation, device="cpu", benchmark="nb101"):
    model = model.to(device)
    lambdas = []
    for c in dataset_classes.keys():
        x_ntks = dataset_classes[c]
        if class_permutation is not None:
            c = class_permutation[c]
        lam = model_min_eigenvalue_class(model, x_ntks, c, benchmark)
        lambdas.append(lam.cpu().numpy())
    return np.sum(lambdas)