import torch
from torch import nn


class ReLUConvBN(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        """
        ReLU + Conv + BatchNorm
        :param C_in: input channels
        :param C_out: output channels
        :param kernel_size: kernel size
        :param stride: convolution stride
        :param padding: input image padding
        :param affine: batch norm parameter
        """
        super(ReLUConvBN, self).__init__()
        self.op = nn.Sequential(nn.ReLU(inplace=False),
                                nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
                                nn.BatchNorm2d(C_out, affine=affine))

    def forward(self, x):
        return self.op(x)


class ReLUDepthwiseConvBN(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        """
        ReLU + DepthwiseConv + BatchNorm
        :param C_in: input channels
        :param C_out: output channels
        :param kernel_size: kernel size
        :param stride: convolution stride
        :param padding: input image padding
        :param affine: batch norm parameter
        """
        super(ReLUDepthwiseConvBN, self).__init__()
        self.op = nn.Sequential(nn.ReLU(inplace=False),
                                nn.Conv2d(C_in, C_in, kernel_size, stride=stride, padding=padding, bias=False,
                                          groups=C_in),
                                nn.Conv2d(C_in, C_out, kernel_size=1, stride=stride, padding=padding, bias=False),
                                nn.BatchNorm2d(C_out, affine=affine))

    def forward(self, x):
        return self.op(x)


class ReLUTwoConvBN(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        """
        ReLU + Conv kx1 + Conv 1*k + BatchNorm
        :param C_in: input channels
        :param C_out: output channels
        :param kernel_size: kernel size
        :param stride: convolution stride
        :param padding: input image padding
        :param affine: batch norm parameter
        """
        super(ReLUTwoConvBN, self).__init__()
        self.op = nn.Sequential(nn.ReLU(inplace=False),
                                nn.Conv2d(C_in, C_in, kernel_size=(kernel_size, 1), stride=stride, padding=padding,
                                          bias=False),
                                nn.Conv2d(C_in, C_in, kernel_size=(1, kernel_size), stride=stride, padding=padding,
                                          bias=False),
                                nn.BatchNorm2d(C_out, affine=affine))

    def forward(self, x):
        return self.op(x)


class FactorizedReduce(nn.Module):

    def __init__(self, C_in, C_out, stride, affine):
        super(FactorizedReduce, self).__init__()
        self.stride = stride
        self.C_in = C_in
        self.C_out = C_out
        self.relu = nn.ReLU(inplace=False)
        if stride == 2:
            # assert C_out % 2 == 0, 'C_out : {:}'.format(C_out)
            C_outs = [C_out // 2, C_out - C_out // 2]
            self.convs = nn.ModuleList()
            for i in range(2):
                self.convs.append(
                    nn.Conv2d(
                        C_in, C_outs[i], 1, stride=stride, padding=0, bias=not affine
                    )
                )
            self.pad = nn.ConstantPad2d((0, 1, 0, 1), 0)
        elif stride == 1:
            self.conv = nn.Conv2d(
                C_in, C_out, 1, stride=stride, padding=0, bias=not affine
            )
        else:
            raise ValueError("Invalid stride : {:}".format(stride))
        self.bn = nn.BatchNorm2d(C_out, affine=affine)

    def forward(self, x):
        if self.stride == 2:
            x = self.relu(x)
            y = self.pad(x)
            out = torch.cat([self.convs[0](x), self.convs[1](y[:, :, 1:, 1:])], dim=1)
        else:
            out = self.conv(x)
        out = self.bn(out)
        return out

    def extra_repr(self):
        return "C_in={C_in}, C_out={C_out}, stride={stride}".format(**self.__dict__)

class Zero(nn.Module):
        def __init__(self, C_in, C_out, stride):
            super(Zero, self).__init__()
            self.C_in = C_in
            self.C_out = C_out
            self.stride = stride
            self.is_zero = True

        def forward(self, x):
            if self.C_in == self.C_out:
                if self.stride == 1:
                    return x.mul(0.0)
                else:
                    return x[:, :, :: self.stride, :: self.stride].mul(0.0)
            else:
                shape = list(x.shape)
                shape[1] = self.C_out
                zeros = x.new_zeros(shape, dtype=x.dtype, device=x.device)
                return zeros

class Pooling(nn.Module):
    def __init__(
        self, C_in, C_out, stride, mode, affine=True):
        super(Pooling, self).__init__()
        if C_in == C_out:
            self.preprocess = None
        else:
            self.preprocess = ReLUConvBN(
                C_in, C_out, 1, 1, 0, affine)
        if mode == "avg":
            self.op = nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False)
        elif mode == "max":
            self.op = nn.MaxPool2d(3, stride=stride, padding=1)
        else:
            raise ValueError("Invalid mode={:} in POOLING".format(mode))

    def forward(self, inputs):
        if self.preprocess:
            x = self.preprocess(inputs)
        else:
            x = inputs
        return self.op(x)