"""
File: tracker_tensorboard.py
Author: Jake Tuero (tuero@ualberta.ca)
Date: April 26, 2021

Callback for tensorboard metric logging
"""

import os
import gin
from torch import functional
from torch.utils.tensorboard import SummaryWriter
from ptu.callbacks.callback_base import Callback
from ptu.util.types import MetricFramework, MetricType, Mode


@gin.configurable
class TrackerTensorboard(Callback):
    def __init__(self, tensorboard_dir: str, experiment: str):
        """Logs various metrics using the tensorboard API.
        To log a metric, store a MetricsItem in the trainer.metrics_buffer, and it will be pushed
        at the next opportunity. See types.MetricsItem for details.
        The initializer arguments can be set through gin.

        Args:
            tensorboard_dir: The base directory to store tensorboard metrics, relative to the main script
            experiment: Experiment string name which will be used as a subfolder in tensorboard_dir
        """
        # Check if tensorboard log dir exists, create if not
        if not os.path.exists(tensorboard_dir):
            os.makedirs(tensorboard_dir)

        # Set types of writers
        path_train = os.path.join(tensorboard_dir, experiment, "train")
        path_val = os.path.join(tensorboard_dir, experiment, "val")
        path_test = os.path.join(tensorboard_dir, experiment, "test")
        self.writers = {
            Mode.train: SummaryWriter(path_train),
            Mode.val: SummaryWriter(path_val),
            Mode.test: SummaryWriter(path_test),
        }
        # Mapping of tensorboard calls
        self.type_map = {
            MetricType.scalar: "add_scalar",
            MetricType.scalars: "add_scalars",
            MetricType.image: "add_image",
            MetricType.images: "add_images",
        }

    def tensorboard_out(self):
        """Pulls metrics from buffer, and buffer cleared.
        Called at every callback point
        """
        for metric_item in self.trainer.metrics_buffer:
            if metric_item.framework != MetricFramework.tensorboard:
                continue
            # Check unknown metric type
            if metric_item.metric_type not in self.type_map:
                raise NotImplementedError
            # Track metric
            getattr(self.writers[metric_item.mode], self.type_map[metric_item.metric_type])(*metric_item.metric_info)

        # Clear buffer items
        self.trainer.metrics_buffer = [
            m for m in self.trainer.metrics_buffer if m.framework != MetricFramework.tensorboard
        ]

    def flush_writers(self) -> None:
        for writer in self.writers.values():
            writer.flush()

    def close_writers(self) -> None:
        for writer in self.writers.values():
            writer.close()

    def begin_fit(self) -> bool:
        self.tensorboard_out()
        return True

    def after_fit(self) -> bool:
        self.tensorboard_out()
        return True

    def begin_test(self) -> bool:
        self.tensorboard_out()
        return True

    def after_test(self) -> bool:
        self.tensorboard_out()
        return True

    def begin_epoch(self) -> bool:
        self.tensorboard_out()
        return True

    def after_epoch(self) -> bool:
        self.tensorboard_out()
        self.flush_writers()
        return True

    def begin_episode(self) -> bool:
        self.tensorboard_out()
        return True

    def after_episode(self) -> bool:
        self.tensorboard_out()
        self.flush_writers()
        return True

    def begin_train_step(self) -> bool:
        self.tensorboard_out()
        return True

    def after_train_step(self) -> bool:
        self.tensorboard_out()
        return True

    def begin_val_step(self) -> bool:
        self.tensorboard_out()
        return True

    def after_val_step(self) -> bool:
        self.tensorboard_out()
        return True

    def begin_batch(self) -> bool:
        self.tensorboard_out()
        return True

    def after_batch(self) -> bool:
        self.tensorboard_out()
        return True

    def after_backward(self) -> bool:
        self.tensorboard_out()
        return True
