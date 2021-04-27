# File: network_util.py
# Author: Jake Tuero (tuero@ualberta.ca)
# Date: April 26, 2021
#
# Utility functions to create a network from a config
# Handles simple networks (linear, conv, conv-to-linear, linear-to-conv)
# Will support more complicated variants as needed, but create network as normal if not sure.

from typing import Any
import numpy as np
import torch.nn as nn

from ptu.util.types import LayerType


# Create a network from a config (list of layer type + dict attributes)
# in_channels and in_size descripe the input tensor shape
# (so we can automatically figure out the shapes of next layers)
def create_network(config, in_channels, in_size):
    layers = []
    curr_channels = in_channels
    size = in_size
    for layer_type, layer_attr in config:
        layer, curr_channels, size = _process_layer(layer_type, layer_attr, curr_channels, size)
        layers.append(layer)
    return nn.Sequential(*layers), curr_channels, size


# Create the network backwards in reference to the given config
# This is most commonly used for symetric networks in terms of encoder/decoder networks,
# so that I don't have to worry about 2 separate configs (one for each)
def create_reverse_network(config, out_channels, out_size):
    layers = []
    curr_channels = out_channels
    size = out_size
    rev_config = [list(c) for c in config]
    # Ensure we do not have activation or batchnorm/dropout on last layer (usually this is reconstruction)
    rev_config[0][1]["act_func"] = nn.Identity
    rev_config[0][1].pop("bn", None)
    rev_config[0][1].pop("p_drop", None)
    for layer_type, layer_attr in config:
        layer, curr_channels, size = _process_layer_reversed(layer_type, layer_attr, curr_channels, size)
        layers.insert(0, layer)
    return nn.Sequential(*layers), curr_channels, size


def _process_layer(layer_type, layer_attr, in_channels, size):
    _layer_map = {LayerType.flatten: process_flat, LayerType.linear: process_linear, LayerType.conv2d: process_conv}
    return _layer_map[layer_type](layer_attr, in_channels, size)


def _process_layer_reversed(layer_type, layer_attr, in_channels, size):
    _layer_map = {
        LayerType.flatten: process_flat_reversed,
        LayerType.linear: process_linear_reversed,
        LayerType.conv2d: process_convT,
    }
    return _layer_map[layer_type](layer_attr, in_channels, size)


# --------------------------------------------
# Functions to process a layer type, takes in
# the layer attributes as well as current
# tensor shapes.
# Output sizes are also calculated.
# --------------------------------------------


def process_conv(layer_attr, curr_channels, curr_size):
    # Layer config
    c, k, p, s = layer_attr["filters"], layer_attr["kernel_size"], layer_attr["padding"], layer_attr["stride"]
    act_func = nn.ReLU if "act_func" not in layer_attr else layer_attr["act_func"]
    bn = True if "bn" in layer_attr else False
    pool = None if "pool" not in layer_attr else layer_attr["pool"]
    p_drop = 0 if "p_drop" not in layer_attr else layer_attr["p_drop"]
    # Create layer
    layer = create_convblock(
        curr_channels, c, kernel_size=k, padding=p, stride=s, act_func=act_func, bn=bn, p_drop=p_drop, pool=pool
    )
    # Update sizes (including if pooling is done)
    size = conv2d_size_out(curr_size, k, p, s)
    if pool is not None:
        size = conv2d_size_out(size, 2, 0, 2)
    return layer, c, size


def process_convT(layer_attr, curr_channels, curr_size):
    # Layer config
    c, k, p, s = layer_attr["filters"], layer_attr["kernel_size"], layer_attr["padding"], layer_attr["stride"]
    act_func = nn.ReLU if "act_func" not in layer_attr else layer_attr["act_func"]
    bn = True if "bn" in layer_attr else False
    pool = None if "pool" not in layer_attr else layer_attr["pool"]
    p_drop = 0 if "p_drop" not in layer_attr else layer_attr["p_drop"]
    # Create layer
    layer = create_convTblock(
        c, curr_channels, kernel_size=k, padding=p, stride=s, act_func=act_func, bn=bn, p_drop=p_drop, pool=pool
    )
    # Update sizes (including if pooling is done)
    if pool is not None:
        curr_size *= 2
    size = conv2dT_size_out(curr_size, k, p, s)
    return layer, c, size


def process_flat(layer_attr, curr_channels, curr_size):
    return nn.Flatten(), -1, curr_size * curr_size * curr_channels


def process_flat_reversed(layer_attr, curr_channels, curr_size):
    size = int(np.sqrt(curr_size))
    return nn.Unflatten(1, (curr_channels, size, size)), curr_channels, curr_channels * size * size


def process_linear(layer_attr, curr_channels, curr_size):
    # Layer config
    out_size = layer_attr["size"]
    act_func = nn.ReLU if "act_func" not in layer_attr else layer_attr["act_func"]
    bn = True if "bn" in layer_attr else False
    p_drop = 0 if "p_drop" not in layer_attr else layer_attr["p_drop"]
    layer = create_linearblock(curr_size, out_size, act_func=act_func, bn=bn, p_drop=p_drop)
    # Size is already given as output size
    return layer, curr_channels, out_size


def process_linear_reversed(layer_attr, curr_channels, curr_size):
    # Layer config
    out_size = layer_attr["size"]
    act_func = nn.ReLU if "act_func" not in layer_attr else layer_attr["act_func"]
    bn = True if "bn" in layer_attr else False
    p_drop = 0 if "p_drop" not in layer_attr else layer_attr["p_drop"]
    layer = create_linearblock(out_size, curr_size, act_func=act_func, bn=bn, p_drop=p_drop)
    # Size is already given as output size
    return layer, curr_channels, out_size


# --------------------------------------------
# Functions to create an individual layer block,
# given all layer attributse
# --------------------------------------------


def create_linearblock(
    nodes_in: int,
    nodes_out: int,
    *args,
    act_func: nn.Module = nn.ReLU,
    bn: bool = False,
    p_drop: float = 0,
    **kwargs: Any
):
    modules = [nn.Linear(nodes_in, nodes_out, *args, **kwargs)]
    if bn:
        modules.append(nn.BatchNorm1d(nodes_out))
    modules.append(act_func())
    if p_drop > 0:
        modules.append(nn.Dropout(p_drop))
    return nn.Sequential(*modules)


def create_convblock(
    channels_in: int,
    channels_out: int,
    *args: Any,
    act_func: nn.Module = nn.ReLU,
    bn: bool = False,
    p_drop: float = 0,
    pool: nn.Module = None,
    **kwargs: Any
):
    modules = [nn.Conv2d(channels_in, channels_out, *args, **kwargs)]
    if pool is not None:
        modules.append(pool)
    if bn:
        modules.append(nn.BatchNorm2d(channels_out))
    modules.append(act_func())
    if p_drop > 0:
        modules.append(nn.Dropout2d(p_drop))
    return nn.Sequential(*modules)


def create_convTblock(
    channels_in: int,
    channels_out: int,
    *args: Any,
    act_func: nn.Module = nn.ReLU,
    bn: bool = False,
    p_drop: float = 0,
    pool: nn.Module = None,
    **kwargs: Any
):
    modules = []
    if pool is not None:
        modules.append(nn.Upsample(scale_factor=2, mode="nearest"))
    modules.append(nn.ConvTranspose2d(channels_in, channels_out, *args, **kwargs))
    if bn:
        modules.append(nn.BatchNorm2d(channels_out))
    modules.append(act_func())
    if p_drop > 0:
        modules.append(nn.Dropout2d(p_drop))
    return nn.Sequential(*modules)


def conv2d_size_out(size, kernel_size=3, padding=0, stride=1):
    return (size + (2 * padding) - kernel_size) // stride + 1


def conv2dT_size_out(size, kernel_size=3, padding=0, stride=1):
    return (size - 1) * stride - (2 * padding) + kernel_size
