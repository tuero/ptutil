"""
File: grad_clipping.py
Author: Jake Tuero (tuero@ualberta.ca)
Date: April 26, 2021

Callback to clip the gradients
"""

import gin
import torch.nn as nn
from ptu.callbacks.callback_base import Callback


@gin.configurable
class GradClipping(Callback):
    def __init__(self, clip_value: int = 1.0):
        """Clips the gradient of the network parameters stored in trainer.grad_clip_buffer.
        Generally speaking, the network's get stored in that buffer as a result of calling
        model.optimization_step() with the model as input.
        The clip_value can be set from a gin config.

        Args:
            clip_value: The value to clip gradients at
        """
        self.clip_value = clip_value

    def _clip(self) -> None:
        for net in self.trainer.grad_clip_buffer:
            nn.utils.clip_grad_norm_(net.parameters(), self.clip_value)
        self.trainer.grad_clip_buffer.clear()

    def after_backward(self) -> bool:
        """Gradient gets clipped after backward pass"""
        self._clip()
        return True
