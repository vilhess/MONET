import random

import torch
from torch import nn

import numpy as np

from search_spaces.nas_bench_201.NASBench201Node import NASBench201Cell
from utils.model_trainer import ModelTrainer
from utils.operations import FactorizedReduce, ReLUConvBN, Zero, Pooling
from utils.pytorch_dataset import RadarDavaDataset

OPERATIONS = {"nor_conv_1x1": lambda C_in, C_out, stride, affine: ReLUConvBN(C_in, C_out, 1, stride, "same", affine),
              "nor_conv_3x3": lambda C_in, C_out, stride, affine: ReLUConvBN(C_in, C_out, 3, stride, "same", affine),
              "none": lambda C_in, C_out, stride, affine: Zero(C_in, C_out, stride),
              "avg_pool_3x3": lambda C_in, C_out, stride, affine: Pooling(C_in, C_out, stride, "avg", affine),
              "skip_connect": lambda C_in, C_out, stride, affine: nn.Identity() if stride == 1 and C_in == C_out
              else FactorizedReduce(C_in, C_out, stride, affine),
              }


class NASBench201NetworkCell(nn.Module):

    def __init__(self, cell_str, C_in, C_out, n_vertices=4):
        super().__init__()
        self.n_vertices = n_vertices
        self.C_in = C_in
        self.C_out = C_out
        self.matrix = None
        self.build(cell_str)

    def build(self, cell_str):
        connexions = cell_str.split("+")
        matrix = nn.ModuleList()
        for i, connexion in enumerate(connexions):
            row = nn.ModuleList()
            list_c = [x for x in connexion.split("|") if x]
            for j, operation in enumerate(list_c):
                op, idx = operation.split('~')
                if i == 0:
                    row.append(OPERATIONS[op](self.C_in, self.C_out, stride=1, affine=True))
                else:
                    row.append(OPERATIONS[op](self.C_out, self.C_out, stride=1, affine=True))
            matrix.append(row)
        self.matrix = matrix

    def forward(self, x):
        for i in range(self.n_vertices - 1):
            current_op = []
            for j in range(i + 1):
                # print(f"{i}, {j}: {self.matrix[i][j]}")
                element = self.matrix[i][j]
                current_op.append(element(x))
            x = torch.stack(current_op, dim=0).sum(dim=0)
        return x


class NASBench201Model(nn.Module):

    def __init__(self, cell_str, input_size, input_depth):
        super().__init__()
        self.cell_str = cell_str
        self.backbone = self.build_backbone(input_size, input_depth)

    def build_backbone(self, input_size, input_depth):
        pass

    def forward(self):
        pass


class NASBench201UNet(NASBench201Model):

    def __init__(self, cell_str, input_size, input_depth):
        self.C = 16
        self.N = 5
        self.layer_channels = [self.C] * self.N + [self.C * 2] + [self.C * 2] * self.N + [self.C * 4] + [
            self.C * 4] * self.N
        self.layer_reductions = [False] * self.N + [True] + [False] * self.N + [True] + [False] * self.N
        super().__init__(cell_str, input_size, input_depth)

    def build_backbone(self, input_size, input_depth):
        self.first_conv = ReLUConvBN(input_depth, self.C, 3, 1, "same", True)
        self.encoder = nn.ModuleList()
        for lc, reduction in zip(self.layer_channels, self.layer_reductions):
            if not reduction:
                c = NASBench201NetworkCell(self.cell_str, C_in=lc, C_out=lc, n_vertices=4)
                self.encoder.append(c)
            else:
                c = ReLUConvBN(lc // 2, lc, 3, 2, 1, True)
                self.encoder.append(c)

        self.bottom_conv = ReLUConvBN(self.layer_channels[-1], self.layer_channels[-1], 3, 1, "same", True)

        self.decoder = nn.ModuleList()
        for lc, reduction in zip(reversed(self.layer_channels), reversed(self.layer_reductions)):
            if not reduction:
                c = NASBench201NetworkCell(self.cell_str, C_in=lc, C_out=lc, n_vertices=4)
                self.decoder.append(c)
            else:
                c = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
                self.decoder.append(c)
                c = ReLUConvBN(lc, lc // 2, 3, 1, 1, True)
                self.decoder.append(c)

        self.last_conv = nn.Conv2d(self.layer_channels[0], input_depth, 1, 1, "same")

    def forward(self, x):
        x = self.first_conv(x)

        encoder_tensors = []
        for i, mod in enumerate(self.encoder):
            x = mod(x)
            # print(f"{i} : {x.shape}")
            encoder_tensors.append(x)

        x = self.bottom_conv(x)

        for i, mod in enumerate(self.decoder):
            x = mod(x)
            if isinstance(self.decoder[i - 1], nn.Upsample):
                x = torch.add(x, list(reversed(encoder_tensors))[i])
        x = self.last_conv(x)
        x = nn.Sigmoid()(x)
        return x

class NASBench201UNet_NTK(NASBench201UNet):

    def __init__(self, cell_str, input_size, input_depth):
        super().__init__(cell_str, input_size, input_depth)

    def build_backbone(self, input_size, input_depth):
        super().build_backbone(input_size, input_depth)
        latent_space_dim1 = input_size // (2 * np.sum(self.layer_reductions))
        latent_space_dim2 = np.max(self.layer_channels)
        self.dense_ntk = nn.Linear(in_features=latent_space_dim1 * latent_space_dim1 * latent_space_dim2,
                                   out_features=10)

    def forward(self, x):
        x = self.first_conv(x)
        encoder_tensors = []
        for i, mod in enumerate(self.encoder):
            x = mod(x)
            encoder_tensors.append(x)
        x_bottom = self.bottom_conv(x)
        x_for_ntk = x_bottom.view(x_bottom.shape[0], -1)
        x_for_ntk = self.dense_ntk(x_for_ntk)
        return x_for_ntk

if __name__ == '__main__':
    """
    Setting project environment
    """
    PATH = "/projets/users/T0259728/radarconf24/train_bth/mat"  # for DLAB
    # PATH = "C:/Users/T0259728/projets/data"  # for windows
    # torch.manual_seed(308)
    tl.set_backend("pytorch")
    """Creating model trainer"""
    # region Model creation
    dataset = RadarDavaDataset(root_dir=PATH, has_distance=True, batch_size=8)
    train_loader, test_loader, val_loader = dataset.generate_loaders(test_split=0.8, val_split=0.8)
    trainer = ModelTrainer(params_path="../../utils/params.json", disable_tqdm=False)
    trainer.set_data(train_loader, val_loader)
    trainer.set_test_data(test_loader)
    # endregion

    cell = NASBench201Cell(4)
    while not cell.is_complete():
        actions = cell.get_action_tuples()
        act = random.choice(actions)
        cell.play_action(*act)
    print(cell.to_str())
    c = NASBench201NetworkCell(cell.to_str(), C_in=1, C_out=4, n_vertices=cell.n_vertices)
    unet = NASBench201UNet_NTK(cell.to_str(), input_size=128, input_depth=1)
    n_params = sum(p.numel() for p in unet.parameters())
    print(f"Model has {n_params} params")
    device = torch.device("cuda:0")
    if device is not None:
        trainer.device = torch.device(device)
        unet.to(trainer.device)
    trainer.set_model(unet)
    trainer.set_parameters(trainer.params_path)
    unet(next(iter(train_loader))[0].to(device))
    x, y = next(iter(train_loader))
    print("Model insitializaed")
    ntk = NTK(unet)
    ntkk = ntk.compute_ntk(x.to(device), x.to(device))
    min_ev = compute_min_eigenvalue(ntkk)
    print(f"Min eigen value: {min_ev[0]}")
