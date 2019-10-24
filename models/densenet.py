import torch
import torch.nn as nn
from tensorboardX import SummaryWriter


class BottleNeck(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super(BottleNeck, self).__init__()
        inner_channel = 4 * growth_rate

        self.bottle_neck = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, inner_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(inner_channel),
            nn.ReLU(),
            nn.Conv2d(inner_channel, growth_rate, kernel_size=3, padding=1, bias=False)
        )

    def forward(self, x):
        return torch.cat([x, self.bottle_neck(x)], 1)


class Transition(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Transition, self).__init__()

        self.down_sample = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.AvgPool2d(2, stride=2)
        )

    def forward(self, x):
        return self.down_sample(x)


class DenseNet(nn.Module):
    def __init__(self, block, nblocks, growth_rate=12, reduction=0.5, num_classes=100):
        super(DenseNet, self).__init__()

        self.growth_rate = growth_rate

        inner_channels = 2 * growth_rate

        self.conv1 = nn.Conv2d(3, inner_channels, kernel_size=3, padding=1, bias=False)

        self.features = nn.Sequential()

        for index in range(len(nblocks) - 1):
            self.features.add_module("dense_block_layer_{}".format(index),
                                     self._make_dense_layers(block, inner_channels, nblocks[index]))
            inner_channels += growth_rate * nblocks[index]
            out_channels = int(reduction * inner_channels)
            self.features.add_module("transition_layer_{}".format(index), Transition(inner_channels, out_channels))
            inner_channels = out_channels

        self.features.add_module("dense_block_layer_{}".format(len(nblocks) - 1),
                                 self._make_dense_layers(block, inner_channels, nblocks[len(nblocks) - 1]))
        inner_channels += growth_rate * nblocks[len(nblocks) - 1]
        self.features.add_module("bn", nn.BatchNorm2d(inner_channels))
        self.features.add_module("relu", nn.ReLU())

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.linear = nn.Linear(inner_channels, num_classes)

    def _make_dense_layers(self, block, in_channels, nblocks):
        dense_block = nn.Sequential()
        for index in range(nblocks):
            dense_block.add_module("bottle_neck_layer_{}".format(index), block(in_channels, self.growth_rate))
            in_channels += self.growth_rate
        return dense_block

    def forward(self, x):
        out = self.conv1(x)
        out = self.features(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def densenet(num_layers, num_classes):
    if num_layers == 121:
        return densenet121(num_classes)
    elif num_layers == 169:
        return densenet169(num_classes)
    elif num_layers == 201:
        return densenet201(num_classes)
    elif num_layers == 264:
        return densenet201(num_classes)
    elif num_layers == 161:
        return densenet161(num_classes)


def densenet121(num_classes):
    return DenseNet(BottleNeck, [6, 12, 24, 16], growth_rate=32, num_classes=num_classes)


def densenet169(num_classes):
    return DenseNet(BottleNeck, [6, 12, 32, 32], growth_rate=32, num_classes=num_classes)


def densenet201(num_classes):
    return DenseNet(BottleNeck, [6, 12, 48, 32], growth_rate=32, num_classes=num_classes)


def densenet264(num_classes):
    return DenseNet(BottleNeck, [6, 12, 64, 48], growth_rate=32, num_classes=num_classes)


def densenet161(num_classes):
    return DenseNet(BottleNeck, [6, 12, 36, 24], growth_rate=48, num_classes=num_classes)


if __name__ == '__main__':
    model = densenet121(10)
    dummy_input = torch.autograd.Variable(torch.rand(1, 3, 32, 32))
    out = model(dummy_input)
    #    print(out.size())
    with SummaryWriter(comment='densenet121') as w:
        w.add_graph(model, (dummy_input,))