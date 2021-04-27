# File: types.py
# Author: Jake Tuero (tuero@ualberta.ca)
# Date: April 26, 2021
#
# Various types used throughout this library

from typing import Any
from enum import Enum
from dataclasses import dataclass
import torch.optim as optim


# Types of supported metric frameworks
class MetricFramework(Enum):
    tensorboard = 1
    comet = 2


# Types of metrics which we can track
class MetricType(Enum):
    scalar = 1
    scalars = 2
    image = 3
    images = 4


# Main modes the trainer can be in
class Mode(Enum):
    train = 1
    val = 2
    test = 3


# Layers recognized by the network creator
class LayerType(Enum):
    linear = 1
    conv2d = 2
    convT2d = 3
    flatten = 4


# Data class for things which are logged
@dataclass
class LoggingItem:
    level: str = "INFO"
    msg: str = ""


# Data class for metrics which are tracked
@dataclass
class MetricsItem:
    framework: MetricFramework
    metric_type: MetricType
    mode: Mode
    metric_info: Any


# Data class for optimizer + scheduler
# Need clean way to handle with/without scheduler, so we
#   just assume we always have a scheduler (use step with 1 multiplier)
@dataclass
class OptimizerInfo:
    optimizer: optim
    scheduler: optim.lr_scheduler = None
    interval: str = "epoch"

    def __post_init__(self):
        assert self.interval in ["epoch", "step"]
        if self.scheduler is None:
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=1.0, last_epoch=-1)
