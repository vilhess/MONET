import torch
import torch.nn as nn
from helpers import calculate_padding


# import torchvision

class DownBlock(nn.Module):
    """
    Double Conv and reduces input size by 2
    """

    def __init__(self, in_channels, out_channels, input_size, kernel_size):
        super(DownBlock, self).__init__()
        # Strided conv that reduces input size by 2
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=2,
                               padding=calculate_padding(input_size, input_size // 2, kernel_size, 2))
        self.bn1 = nn.BatchNorm2d(num_features=out_channels)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size,
                               padding=calculate_padding(input_size // 2, input_size // 2, kernel_size, 1))
        self.bn2 = nn.BatchNorm2d(num_features=out_channels)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        xa = self.conv1(x)
        xb = self.bn1(xa)
        x = self.relu1(xb)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        return x


class UpBlock(nn.Module):
    """
    Upscale followed by double conv.
    """

    def __init__(self, in_channels, out_channels, input_size, kernel_size):
        super(UpBlock, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        input_size *= 2
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                               padding=calculate_padding(input_size, input_size, kernel_size, stride=1))
        self.bn1 = nn.BatchNorm2d(num_features=out_channels)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size,
                               padding=calculate_padding(input_size, input_size, kernel_size, stride=1))
        self.bn2 = nn.BatchNorm2d(num_features=out_channels)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        x = self.upsample(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        return x


class Add_and_conv(nn.Module):
    """
    Adds two tensors and performs a convolution
    """

    def __init__(self, in_channels, out_channels, input_size, kernel_size):
        super(Add_and_conv, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                              padding=calculate_padding(input_size, input_size, kernel_size, stride=1))

    def forward(self, x1, x2):
        x = torch.add(x1, x2)
        x = self.conv(x)
        return x


class UNet(nn.Module):

    def __init__(self, input_size):
        super(UNet, self).__init__()
        self.input_size = input_size
        tensor_size = input_size  # 256,256

        self.down_block_1 = DownBlock(in_channels=1, out_channels=16, input_size=tensor_size, kernel_size=3)
        tensor_size //= 2  # 128,128
        self.down_block_2 = DownBlock(in_channels=16, out_channels=32, input_size=tensor_size, kernel_size=3)
        tensor_size //= 2  # 64,64
        self.down_block_3 = DownBlock(in_channels=32, out_channels=64, input_size=tensor_size, kernel_size=3)
        tensor_size //= 2  # 32,32

        self.up_block_1 = UpBlock(in_channels=64, out_channels=32, input_size=tensor_size, kernel_size=3)
        tensor_size *= 2  # 64,64
        self.add_1 = Add_and_conv(in_channels=32, out_channels=32, input_size=tensor_size, kernel_size=3)

        self.up_block_2 = UpBlock(in_channels=32, out_channels=16, input_size=tensor_size, kernel_size=3)
        tensor_size *= 2  # 128,128
        self.add_2 = Add_and_conv(in_channels=16, out_channels=16, input_size=tensor_size, kernel_size=3)

        self.up_block_3 = UpBlock(in_channels=16, out_channels=8, input_size=tensor_size, kernel_size=3)
        tensor_size *= 2  # 256,256

        self.final_conv = nn.Conv2d(in_channels=8, out_channels=1, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x1 = self.down_block_1(x)  # Channels = 16, dim = 128,128
        x2 = self.down_block_2(x1)  # Channels = 32, dim = 64,64
        x3 = self.down_block_3(x2)  # Channels = 64, dim = 32,32

        x4 = self.up_block_1(x3)  # Channels = 32, dim = 64,64
        x4 = self.add_1(x2, x4)

        x5 = self.up_block_2(x4)  # Channels = 16, dim = 128,128
        x5 = self.add_2(x1, x5)

        x6 = self.up_block_3(x5)  # Channels = 1, dim = 256,256

        x7 = self.final_conv(x6)
        x8 = nn.Sigmoid()(x7)
        return x8

class UNetLite(nn.Module):

    def __init__(self, input_size):
        super(UNetLite, self).__init__()
        self.input_size = input_size
        tensor_size = input_size  # 256,256

        self.down_block_1 = DownBlock(in_channels=1, out_channels=5, input_size=tensor_size, kernel_size=3)
        tensor_size //= 2  # 128,128
        self.down_block_2 = DownBlock(in_channels=5, out_channels=10, input_size=tensor_size, kernel_size=3)
        tensor_size //= 2  # 64,64
        self.down_block_3 = DownBlock(in_channels=10, out_channels=20, input_size=tensor_size, kernel_size=3)
        tensor_size //= 2  # 32,32

        self.up_block_1 = UpBlock(in_channels=20, out_channels=10, input_size=tensor_size, kernel_size=3)
        tensor_size *= 2  # 64,64
        self.add_1 = Add_and_conv(in_channels=10, out_channels=10, input_size=tensor_size, kernel_size=3)

        self.up_block_2 = UpBlock(in_channels=10, out_channels=5, input_size=tensor_size, kernel_size=3)
        tensor_size *= 2  # 128,128
        self.add_2 = Add_and_conv(in_channels=5, out_channels=5, input_size=tensor_size, kernel_size=3)

        self.up_block_3 = UpBlock(in_channels=5, out_channels=5, input_size=tensor_size, kernel_size=3)
        tensor_size *= 2  # 256,256

        self.final_conv = nn.Conv2d(in_channels=5, out_channels=1, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x1 = self.down_block_1(x)  # Channels = 16, dim = 128,128
        x2 = self.down_block_2(x1)  # Channels = 32, dim = 64,64
        x3 = self.down_block_3(x2)  # Channels = 64, dim = 32,32

        x4 = self.up_block_1(x3)  # Channels = 32, dim = 64,64
        x4 = self.add_1(x2, x4)

        x5 = self.up_block_2(x4)  # Channels = 16, dim = 128,128
        x5 = self.add_2(x1, x5)

        x6 = self.up_block_3(x5)  # Channels = 1, dim = 256,256

        x7 = self.final_conv(x6)
        x8 = nn.Sigmoid()(x7)
        return x8


class LargeUNet(nn.Module):

    def __init__(self, input_size):
        super(LargeUNet, self).__init__()
        self.input_size = input_size
        tensor_size = input_size  # 256,256

        self.down_block_1 = DownBlock(in_channels=1, out_channels=32, input_size=tensor_size, kernel_size=3)
        tensor_size //= 2  # 128,128
        self.down_block_2 = DownBlock(in_channels=32, out_channels=64, input_size=tensor_size, kernel_size=3)
        tensor_size //= 2  # 64,64
        self.down_block_3 = DownBlock(in_channels=64, out_channels=128, input_size=tensor_size, kernel_size=3)
        tensor_size //= 2  # 32,32

        self.up_block_1 = UpBlock(in_channels=128, out_channels=64, input_size=tensor_size, kernel_size=3)
        tensor_size *= 2  # 64,64
        self.add_1 = Add_and_conv(in_channels=64, out_channels=64, input_size=tensor_size, kernel_size=3)

        self.up_block_2 = UpBlock(in_channels=64, out_channels=32, input_size=tensor_size, kernel_size=3)
        tensor_size *= 2  # 128,128
        self.add_2 = Add_and_conv(in_channels=32, out_channels=32, input_size=tensor_size, kernel_size=3)

        self.up_block_3 = UpBlock(in_channels=32, out_channels=8, input_size=tensor_size, kernel_size=3)
        tensor_size *= 2  # 256,256

        self.final_conv = nn.Conv2d(in_channels=8, out_channels=1, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x1 = self.down_block_1(x)  # Channels = 16, dim = 128,128
        x2 = self.down_block_2(x1)  # Channels = 32, dim = 64,64
        x3 = self.down_block_3(x2)  # Channels = 64, dim = 32,32

        x4 = self.up_block_1(x3)  # Channels = 32, dim = 64,64
        x4 = self.add_1(x2, x4)

        x5 = self.up_block_2(x4)  # Channels = 16, dim = 128,128
        x5 = self.add_2(x1, x5)

        x6 = self.up_block_3(x5)  # Channels = 1, dim = 256,256

        x7 = self.final_conv(x6)
        x8 = nn.Sigmoid()(x7)
        return x8


if __name__ == "__main__":
    model = UNet(256)
    # model.load_state_dict(torch.load("C:/Users/T0259728/projets/unet/saved_model/model.pt"))
    plot_weight_distribution(model)
