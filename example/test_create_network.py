from ptu.network_util import create_network, create_reverse_network
from ptu.util.types import LayerType
from torchinfo import summary
import torch
import torch.nn as nn


network_config_linear = [
    (LayerType.linear, {"size": 64}),
    (LayerType.linear, {"size": 128, "bn": True, "p_drop": 0.5}),
    (LayerType.linear, {"size": 256, "act_func": nn.Tanh}),
    (LayerType.linear, {"size": 10}),
]


network_config_conv = [
    (
        LayerType.conv2d,
        {"filters": 4, "kernel_size": 3, "padding": 1, "stride": 1, "output_padding": 0, "pool": nn.MaxPool2d(2)},
    ),
    (
        LayerType.conv2d,
        {
            "filters": 8,
            "kernel_size": 3,
            "padding": 1,
            "stride": 1,
            "output_padding": 0,
            "p_drop": 0.2,
            "act_func": nn.Tanh,
        },
    ),
    (
        LayerType.conv2d,
        {"filters": 8, "kernel_size": 3, "padding": 1, "stride": 1, "output_padding": 0, "pool": nn.MaxPool2d(2)},
    ),
    (LayerType.flatten, {}),
    (LayerType.linear, {"size": 256}),
    (LayerType.linear, {"size": 10}),
]


class ModelForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.net, curr_channels, size = create_network(network_config_conv, 3, 64)
        print(curr_channels)
        print(size)

    def forward(self, x):
        return self.net(x)


class ModelReversed(nn.Module):
    def __init__(self):
        super().__init__()
        self.net, curr_channels, size = create_reverse_network(network_config_conv, 3, 64)
        print(curr_channels)
        print(size)

    def forward(self, x):
        return self.net(x)


model1 = ModelForward()
rand_data = torch.rand(4, 3, 64, 64)
print(model1)
model_stats1 = summary(model1, input_size=(4, 3, 64, 64), verbose=0)
print(model_stats1)

model2 = ModelReversed()
rand_data = torch.rand(4, 10)
print(model2)
model_stats2 = summary(model2, input_size=(4, 10), verbose=0)
print(model_stats2)
model2.cpu()
print(model2(rand_data).shape)
