"""
File: types.py
Author: Jake Tuero (tuero@ualberta.ca)
Date: April 26, 2021

Various types used throughout this library
"""

from typing import Any
from enum import Enum
from dataclasses import dataclass
import torch.optim as optim


class MetricFramework(Enum):
    """Types of supported metric frameworks"""

    tensorboard = 1
    comet = 2


class MetricType(Enum):
    """Types of metrics which we can track"""

    scalar = 1
    scalars = 2
    image = 3
    images = 4


class Mode(Enum):
    """Main modes the trainer can be in"""

    train = 1
    val = 2
    test = 3


class LayerType(Enum):
    """Layers recognized by the network creator"""

    linear = 1
    conv2d = 2
    convT2d = 3
    flatten = 4


@dataclass
class LoggingItem:
    """Data class for things which are logged"""

    level: str = "INFO"
    msg: str = ""


@dataclass
class MetricsItem:
    """Data class for metrics which are tracked"""

    framework: MetricFramework
    metric_type: MetricType
    mode: Mode
    metric_info: Any


# Data class for optimizer + scheduler
# Need clean way to handle with/without scheduler, so we
#   just assume we always have a scheduler (use step with 1 multiplier)
@dataclass
class OptimizerInfo:
    """Data class for optimizer + scheduler. Need clean way to handle with/without scheduler,
    so we just assume we always have a scheduler (use step with 1 multiplier)
    """

    optimizer: optim
    scheduler: optim.lr_scheduler = None
    interval: str = "epoch"

    def __post_init__(self):
        assert self.interval in ["epoch", "step"]
        if self.scheduler is None:
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=1.0, last_epoch=-1)
