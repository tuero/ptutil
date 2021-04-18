import os
import gin
from torch import functional
from torch.utils.tensorboard import SummaryWriter
from ptu.callbacks.callback_base import Callback
from ptu.util.types import MetricFramework, MetricType, Mode


# Callback for tracking metrics
@gin.configurable
class TrackerTensorboard(Callback):
    def __init__(self, tensorboard_dir, experiment):
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
            MetricType.image: "add_image",
            MetricType.images: "add_images",
        }

    # Pulls metrics from buffer, and buffer cleared
    # Called at every callback point
    def tensorboard_out(self):
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

    def begin_fit(self):
        self.tensorboard_out()
        return True

    def after_fit(self):
        self.tensorboard_out()
        return True

    def begin_validate(self):
        self.tensorboard_out()
        return True

    def after_validate(self):
        self.tensorboard_out()
        return True

    def begin_test(self):
        self.tensorboard_out()
        return True

    def after_test(self):
        self.tensorboard_out()
        return True

    def begin_epoch(self):
        self.tensorboard_out()
        return True

    def after_epoch(self):
        self.tensorboard_out()
        return True

    def begin_train_step(self):
        self.tensorboard_out()
        return True

    def after_train_step(self):
        self.tensorboard_out()
        return True

    def begin_val_step(self):
        self.tensorboard_out()
        return True

    def after_val_step(self):
        self.tensorboard_out()
        return True

    def begin_batch(self):
        self.tensorboard_out()
        return True

    def after_batch(self):
        self.tensorboard_out()
        return True

    def after_loss(self):
        self.tensorboard_out()
        return True

    def after_backward(self):
        self.tensorboard_out()
        return True
