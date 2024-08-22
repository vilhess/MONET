import sys
sys.path.append('/Users/samy/Desktop/NTK4NAS/')

""" CNN for architecture search """
from torch.autograd import Variable
from search_spaces.DARTS.operations import *
from search_spaces.nas_bench_301.NASBench301Node import DARTSNode, DARTSCell
import random
from collections import namedtuple

def drop_path(x, drop_prob, dims=(0,)):
    var_size = [1 for _ in range(x.dim())]
    for i in dims:
        var_size[i] = x.size(i)
    if drop_prob > 0.:
        keep_prob = 1.-drop_prob
        mask = Variable(torch.cuda.FloatTensor(
            *var_size).bernoulli_(keep_prob))
        x.div_(keep_prob)
        x.mul_(mask)
    return x


class Cell(nn.Module):

    def __init__(self, genotype, C_prev_prev, C_prev, C, reduction, reduction_prev):
        super(Cell, self).__init__()
        # print(C_prev_prev, C_prev, C)

        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0)

        if reduction:
            op_names, indices = zip(*genotype.reduce)
            concat = genotype.reduce_concat
        else:
            op_names, indices = zip(*genotype.normal)
            concat = genotype.normal_concat
        self._compile(C, op_names, indices, concat, reduction)

    def _compile(self, C, op_names, indices, concat, reduction):
        assert len(op_names) == len(indices)
        self._steps = len(op_names) // 2
        self._concat = concat
        self.multiplier = len(concat)

        self._ops = nn.ModuleList()
        for name, index in zip(op_names, indices):
            stride = 2 if reduction and index < 2 else 1
            op = OPS[name](C, stride, True)
            self._ops += [op]
        self._indices = indices

    def forward(self, s0, s1, drop_prob):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        for i in range(self._steps):
            h1 = states[self._indices[2 * i]]
            h2 = states[self._indices[2 * i + 1]]
            op1 = self._ops[2 * i]
            op2 = self._ops[2 * i + 1]
            h1 = op1(h1)
            h2 = op2(h2)
            if self.training and drop_prob > 0.:
                if not isinstance(op1, Identity):
                    h1 = drop_path(h1, drop_prob)
                if not isinstance(op2, Identity):
                    h2 = drop_path(h2, drop_prob)
            s = h1 + h2
            states += [s]
        return torch.cat([states[i] for i in self._concat], dim=1)

class AuxiliaryHeadCIFAR(nn.Module):

    def __init__(self, C, num_classes):
        """assuming input size 8x8"""
        super(AuxiliaryHeadCIFAR, self).__init__()
        self.features = nn.Sequential(
            nn.ReLU(inplace=True),
            # image size = 2 x 2
            nn.AvgPool2d(5, stride=3, padding=0, count_include_pad=False),
            nn.Conv2d(C, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 768, 2, bias=False),
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True)
        )
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x.view(x.size(0), -1))
        return x

class NetworkCIFAR(nn.Module):

    def __init__(self, C, num_classes, layers, auxiliary, genotype):
        super(NetworkCIFAR, self).__init__()
        self._layers = layers
        self._auxiliary = auxiliary
        self.drop_path_prob=0

        stem_multiplier = 3
        C_curr = stem_multiplier * C
        self.stem = nn.Sequential(
            nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
            nn.BatchNorm2d(C_curr)
        )

        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = nn.ModuleList()
        reduction_prev = False
        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = Cell(genotype, C_prev_prev, C_prev,
                        C_curr, reduction, reduction_prev)
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, cell.multiplier * C_curr
            if i == 2 * layers // 3:
                C_to_auxiliary = C_prev

        if auxiliary:
            self.auxiliary_head = AuxiliaryHeadCIFAR(
                C_to_auxiliary, num_classes)
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)

    def forward(self, input):
        logits_aux = None
        s0 = s1 = self.stem(input)
        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1, self.drop_path_prob)
            if i == 2 * self._layers // 3:
                if self._auxiliary and self.training:
                    logits_aux = self.auxiliary_head(s1)
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits #, logits_aux


def get_random_node():
    node = DARTSNode((DARTSCell(), DARTSCell()))
    while not node.is_complete():
        av_actions = node.get_action_tuples()
        ac = random.choice(av_actions)
        node.play_action(ac)
    return node

def get_genotype(node):
    normal_cell_genotype = node.state[0].to_genotype()
    reduction_cell_genotype = node.state[1].to_genotype()
    Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')
    genotype_config = Genotype(
        normal=normal_cell_genotype,
        normal_concat=[2, 3, 4, 5],
        reduce=reduction_cell_genotype,
        reduce_concat=[2, 3, 4, 5]
    )
    return genotype_config

if __name__=="__main__":
    node = get_random_node()
    genotype = get_genotype(node)
    model = NetworkCIFAR(3, 10, 5, False, genotype)

    x = torch.rand(10, 3, 32, 32)
    print(model(x).shape)


# class SearchCNN(nn.Module):
#     """ Search CNN model """
#     def __init__(self, C_in, C, n_classes, n_layers, normal_cell, reduction_cell):
#         """
#         Args:
#             C_in: # of input channels
#             C: # of starting model channels
#             n_classes: # of classes
#             n_layers: # of layers
#             n_nodes: # of intermediate nodes in Cell
#             stem_multiplier
#         """
#         super().__init__()
#         self.C_in = C_in
#         self.C = C
#         self.n_classes = n_classes
#         self.n_layers = n_layers


#         C_cur = stem_multiplier * C
#         self.stem = nn.Sequential(
#             nn.Conv2d(C_in, C_cur, 3, 1, 1, bias=False),
#             nn.BatchNorm2d(C_cur)
#         )

#         size = 0
#         for a in range(n_nodes):
#             for b in range(2+a):
#                 size+=1
#         self.normal_cell=normal_cell
#         self.reduction_cell = reduction_cell

#         # for the first cell, stem is used for both s0 and s1
#         # [!] C_pp and C_p is output channel size, but C_cur is input channel size.
#         C_pp, C_p, C_cur = C_cur, C_cur, C

#         self.cells = nn.ModuleList()
#         reduction_p = False
#         for i in range(n_layers):
#             # Reduce featuremap size and double channels in 1/3 and 2/3 layer.
#             if i in [n_layers//3, 2*n_layers//3]:
#                 C_cur *= 2
#                 reduction = True
#                 cell = SearchCell(n_nodes, C_pp, C_p, C_cur, reduction_p, reduction, self.reduction_cell)
#             else:
#                 reduction = False
#                 cell = SearchCell(n_nodes, C_pp, C_p, C_cur, reduction_p, reduction, normal_cell)

#             reduction_p = reduction
#             self.cells.append(cell)
#             C_cur_out = C_cur * n_nodes
#             C_pp, C_p = C_p, C_cur_out

#         self.gap = nn.AdaptiveAvgPool2d(1)
#         self.linear = nn.Linear(C_p, n_classes)

#     def forward(self, x):
#         s0 = s1 = self.stem(x)

#         for cell in self.cells:
#             s0, s1 = s1, cell(s0, s1)

#         out = self.gap(s1)
#         out = out.view(out.size(0), -1) # flatten
#         logits = self.linear(out)

#         return logits

# if __name__=="__main__":

#     n_nodes=4
#     size = 0
#     for a in range(n_nodes):
#         for b in range(2+a):
#             size+=1
#     normal_cell = [random.choice(list(ops.OPS.keys())) for _ in range(size)]
#     reduction_cell = [random.choice(list(ops.OPS.keys())) for _ in range(size)]

#     model = SearchCNN(3, 6, 10, 4, n_nodes=n_nodes, stem_multiplier=3, normal_cell=normal_cell, reduction_cell=reduction_cell)
#     x = torch.rand(1, 3, 32, 32)
#     print(model(x).shape)
