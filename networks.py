import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from mermaidlite import compute_warped_image_multiNC, identity_map_multiN


class Autoencoder(nn.Module):
    def __init__(self, num_layers, channels):
        super(Autoencoder, self).__init__()
        self.num_layers = num_layers
        down_channels = channels[0]
        up_channels = channels[1]
        self.downConvs = nn.ModuleList([])
        self.upConvs = nn.ModuleList([])
        for depth in range(self.num_layers):
            self.downConvs.append(
                nn.Conv2d(
                    down_channels[depth],
                    down_channels[depth + 1],
                    kernel_size=3,
                    padding=1,
                    stride=2,
                )
            )
            self.upConvs.append(
                nn.ConvTranspose2d(
                    up_channels[depth + 1],
                    up_channels[depth],
                    kernel_size=4,
                    padding=1,
                    stride=2,
                )
            )
        self.lastConv = nn.Conv2d(16, 2, kernel_size=3, padding=1)
        torch.nn.init.zeros_(self.lastConv.weight)

    def forward(self, x, y):
        x = torch.cat([x, y], 1)
        skips = []
        for depth in range(self.num_layers):
            skips.append(x)
            x = F.relu(self.downConvs[depth](x))
        for depth in reversed(range(self.num_layers)):
            x = F.relu(self.upConvs[depth](x))
            x = x[:, :, : skips[depth].size()[2], : skips[depth].size()[3]]
        x = self.lastConv(x)
        return x / 10


def tallAE():
    return Autoencoder(
        5,
        np.array(
            [
                [2, 16, 32, 64, 256, 512],
                [16, 32, 64, 128, 256, 512],
            ]
        ),
    )


class Residual(nn.Module):
    def __init__(self, features):
        super(Residual, self).__init__()
        self.bn1 = nn.BatchNorm2d(num_features=features)
        self.bn2 = nn.BatchNorm2d(num_features=features)

        self.conv1 = nn.Conv2d(features, features, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, padding=1)

    def forward(self, x):
        y = F.relu(self.bn1(x))
        y = self.conv1(y)
        y = F.relu(self.bn2(y))
        y = self.conv2(y)
        return y + x


class UNet(nn.Module):
    def __init__(self, num_layers, channels, dimension):
        super(UNet, self).__init__()

        if dimension == 2:
            self.BatchNorm = nn.BatchNorm2d
            self.Conv = nn.Conv2d
            self.ConvTranspose = nn.ConvTranspose2d
        else:
            self.BatchNorm = nn.BatchNorm3d
            self.Conv = nn.Conv3d
            self.ConvTranspose = nn.ConvTranspose3d
        self.num_layers = num_layers
        down_channels = np.array(channels[0])
        up_channels_out = np.array(channels[1])
        up_channels_in = down_channels[1:] + np.concatenate([up_channels_out[1:], [0]])
        self.downConvs = nn.ModuleList([])
        self.upConvs = nn.ModuleList([])
        # self.residues = nn.ModuleList([])
        self.batchNorms = nn.ModuleList(
            [
                self.BatchNorm(num_features=up_channels_out[_])
                for _ in range(self.num_layers)
            ]
        )
        for depth in range(self.num_layers):
            self.downConvs.append(
                self.Conv(
                    down_channels[depth],
                    down_channels[depth + 1],
                    kernel_size=3,
                    padding=1,
                    stride=2,
                )
            )
            self.upConvs.append(
                self.ConvTranspose(
                    up_channels_in[depth],
                    up_channels_out[depth],
                    kernel_size=4,
                    padding=1,
                    stride=2,
                )
            )
            # self.residues.append(
            #    Residual(up_channels_out[depth])
            # )
        self.lastConv = self.Conv(18, dimension, kernel_size=3, padding=1)
        torch.nn.init.zeros_(self.lastConv.weight)

    def forward(self, x, y):
        x = torch.cat([x, y], 1)
        skips = []
        for depth in range(self.num_layers):
            skips.append(x)
            x = F.relu(self.downConvs[depth](x))
        for depth in reversed(range(self.num_layers)):
            x = F.relu(self.upConvs[depth](x))
            x = self.batchNorms[depth](x)

            x = x[:, :, : skips[depth].size()[2], : skips[depth].size()[3]]
            x = torch.cat([x, skips[depth]], 1)
        x = self.lastConv(x)
        return x / 10


def pad_or_crop(x, shape, dimension):
    y = x[:, : shape[1]]
    if x.size()[1] < shape[1]:
        if dimension == 3:
            y = F.pad(y, (0, 0, 0, 0, 0, 0, shape[1] - x.size()[1], 0))
        else:
            y = F.pad(y, (0, 0, 0, 0, shape[1] - x.size()[1], 0))
    assert y.size()[1] == shape[1]

    return y

class UNet2(nn.Module):
    def __init__(self, num_layers, channels, dimension, input_channels=2):
        super(UNet2, self).__init__()
        self.dimension = dimension
        if dimension == 2:
            self.BatchNorm = nn.BatchNorm2d
            self.Conv = nn.Conv2d
            self.ConvTranspose = nn.ConvTranspose2d
            self.avg_pool = F.avg_pool2d
            self.interpolate_mode = "bilinear"
        else:
            self.BatchNorm = nn.BatchNorm3d
            self.Conv = nn.Conv3d
            self.ConvTranspose = nn.ConvTranspose3d
            self.avg_pool = F.avg_pool3d
            self.interpolate_mode = "trilinear"
        self.num_layers = num_layers
        down_channels = np.array(channels[0])
        up_channels_out = np.array(channels[1])
        up_channels_in = down_channels[1:] + np.concatenate([up_channels_out[1:], [0]])
        self.downConvs = nn.ModuleList([])
        self.upConvs = nn.ModuleList([])
        #        self.residues = nn.ModuleList([])
        self.batchNorms = nn.ModuleList(
            [
                self.BatchNorm(num_features=up_channels_out[_])
                for _ in range(self.num_layers)
            ]
        )
        for depth in range(self.num_layers):
            self.downConvs.append(
                self.Conv(
                    down_channels[depth],
                    down_channels[depth + 1],
                    kernel_size=3,
                    padding=1,
                    stride=2,
                )
            )
            self.upConvs.append(
                self.ConvTranspose(
                    up_channels_in[depth],
                    up_channels_out[depth],
                    kernel_size=4,
                    padding=1,
                    stride=2,
                )
            )
        #            self.residues.append(
        #                Residual(up_channels_out[depth])
        #            )
        self.lastConv = self.Conv(18, dimension, kernel_size=3, padding=1)
        torch.nn.init.zeros_(self.lastConv.weight)

    def forward(self, x, y):
        x = torch.cat([x, y], 1)
        skips = []
        for depth in range(self.num_layers):
            skips.append(x)
            y = self.downConvs[depth](F.leaky_relu(x))
            x = y + pad_or_crop(
                self.avg_pool(x, 2, ceil_mode=True), y.size(), self.dimension
            )
            y = F.layer_norm

        for depth in reversed(range(self.num_layers)):
            y = self.upConvs[depth](F.leaky_relu(x))
            x = y + F.interpolate(
                pad_or_crop(x, y.size(), self.dimension),
                scale_factor=2,
                mode=self.interpolate_mode,
                align_corners=False,
            )
            # x = self.residues[depth](x)
            x = self.batchNorms[depth](x)

            x = x[:, :, : skips[depth].size()[2], : skips[depth].size()[3]]
            x = torch.cat([x, skips[depth]], 1)
        x = self.lastConv(x)
        return x / 10



class UNet3(nn.Module):
    def __init__(self, num_layers, channels, dimension):
        super(UNet3, self).__init__()
        self.dimension = dimension
        if dimension == 2:
            self.GroupNorm = nn.GroupNorm2d
            self.Conv = nn.Conv2d
            self.ConvTranspose = nn.ConvTranspose2d
            self.avg_pool = F.avg_pool2d
            self.interpolate_mode = "bilinear"
        else:
            self.BatchNorm = nn.BatchNorm3d
            self.Conv = nn.Conv3d
            self.ConvTranspose = nn.ConvTranspose3d
            self.avg_pool = F.avg_pool3d
            self.interpolate_mode = "trilinear"
        self.num_layers = num_layers
        down_channels = np.array(channels[0])
        up_channels_out = np.array(channels[1])
        up_channels_in = down_channels[1:] + np.concatenate([up_channels_out[1:], [0]])
        self.downConvs = nn.ModuleList([])
        self.upConvs = nn.ModuleList([])
        #        self.residues = nn.ModuleList([])
        self.batchNorms = nn.ModuleList(
            [
                self.BatchNorm(num_features=up_channels_out[_])
                for _ in range(self.num_layers)
            ]
        )
        for depth in range(self.num_layers):
            self.downConvs.append(
                self.Conv(
                    down_channels[depth],
                    down_channels[depth + 1],
                    kernel_size=3,
                    padding=1,
                    stride=2,
                )
            )
            self.upConvs.append(
                self.ConvTranspose(
                    up_channels_in[depth],
                    up_channels_out[depth],
                    kernel_size=4,
                    padding=1,
                    stride=2,
                )
            )
        #            self.residues.append(
        #                Residual(up_channels_out[depth])
        #            )
        self.lastConv = self.Conv(18, dimension, kernel_size=3, padding=1)
        torch.nn.init.zeros_(self.lastConv.weight)

    def forward(self, x, y):
        x = torch.cat([x, y], 1)
        skips = []
        for depth in range(self.num_layers):
            skips.append(x)
            y = self.downConvs[depth](F.leaky_relu(x))
            x = y + pad_or_crop(
                self.avg_pool(x, 2, ceil_mode=True), y.size(), self.dimension
            )
            y = F.layer_norm

        for depth in reversed(range(self.num_layers)):
            y = self.upConvs[depth](F.leaky_relu(x))
            x = y + F.interpolate(
                pad_or_crop(x, y.size(), self.dimension),
                scale_factor=2,
                mode=self.interpolate_mode,
                align_corners=False,
            )
            # x = self.residues[depth](x)
            x = self.batchNorms[depth](x)

            x = x[:, :, : skips[depth].size()[2], : skips[depth].size()[3]]
            x = torch.cat([x, skips[depth]], 1)
        x = self.lastConv(x)
        return x / 10


class TwoStepNet(nn.Module):
    def __init__(self, primaryNet, secondaryNet, primaryNetWeights=None):
        super(TwoSetpNet, self).__init()
        self.primaryNet = primaryNet()
        self.secondaryNet = secondaryNet(input_channels=3)
        if primaryNetWeights:
            self.primaryNet.load_state_dict(primaryNetWeights)
    def setSpacing(self, spacing):
        self.spacing = spacing

    def forward(self, x, y):
        self.approxPhi = primaryNet(x, y)
        resampledy = compute_warped_image_multiNC(y, approxPhi, self.spacing, 1) 
        self.phi = secondaryNet(x, approxPhi, resampledy)
        return self.phi + self.approxPhi



class ScaledTwoStepNet(nn.Module):
    def __init__(self, primaryNet, secondaryNet, scale, primaryNetWeights=None):
        super(ScaledTwoSetpNet, self).__init()
        self.primaryNet = primaryNet()
        self.secondaryNet = secondaryNet()
        self.scale = scale
        if primaryNetWeights:
            self.primaryNet.load_state_dict(primaryNetWeights)

    def forward(self, x, y):
        if scale != 1:
            smolx, smoly = F.av
        else:
            approxPhi = primaryNet(x, y)

        if scale != 1:
            approxPhi = F.interpolate(approxPhi)


def tallUNet(dimension=2, input_channels=2):
    return UNet(
        5,
        [[2, 16, 32, 64, 256, 512], [16, 32, 64, 128, 256]],
        dimension,
    )


def tallishUNet2(dimension=2):
    return UNet2(
        6,
        [[2, 16, 32, 64, 256, 512, 512], [16, 32, 64, 128, 256, 512]],
        dimension,
    )
def tallerUNet2(dimension=2):
    return UNet2(
        7,
        [[2, 16, 32, 64, 256, 512, 512, 512], [16, 32, 64, 128, 256, 512, 512]],
        dimension,
    )


def tallUNet2(dimension=2, input_channels=2):
    return UNet2(
        5,
        np.array([[2, 16, 32, 64, 256, 512], [16, 32, 64, 128, 256]]),
        dimension,
        input_channels=input_channels
    )


class RegisNet(nn.Module):
    def __init__(self):
        super(RegisNet, self).__init__()
        self.conv1 = nn.Conv2d(2, 10, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(12, 10, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(22, 10, kernel_size=5, padding=2)
        self.conv4 = nn.Conv2d(32, 10, kernel_size=5, padding=2)
        self.conv5 = nn.Conv2d(42, 10, kernel_size=5, padding=2)
        self.conv6 = nn.Conv2d(52, 2, kernel_size=5, padding=2)

    def forward(self, x, y):
        x = torch.cat([x, y], 1)

        x = torch.cat([x, F.relu(self.conv1(x))], 1)
        x = torch.cat([x, F.relu(self.conv2(x))], 1)
        x = torch.cat([x, F.relu(self.conv3(x))], 1)
        x = torch.cat([x, F.relu(self.conv4(x))], 1)
        x = torch.cat([x, F.relu(self.conv5(x))], 1)

        return self.conv6(x)


class FCNet(nn.Module):
    def __init__(self, size=28):
        super(FCNet, self).__init__()
        self.size=size
        self.dense1 = nn.Linear(size *size  * 2, 8000)
        self.dense2 = nn.Linear(8000, 3000)
        self.dense3 = nn.Linear(3000, size * size * 2)
        torch.nn.init.zeros_(self.dense3.weight)

    def forward(self, x, y):
        x = torch.reshape(torch.cat([x, y], 1), (-1, 2 * self.size * self.size))
        x = F.relu(self.dense1(x))
        x = F.relu(self.dense2(x))
        x = self.dense3(x)
        x = torch.reshape(x, (-1, 2, self.size, self.size))
        return x
