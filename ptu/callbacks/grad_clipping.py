# File: grad_clipping.py
# Author: Jake Tuero (tuero@ualberta.ca)
# Date: April 26, 2021
#
# Callback to clip the gradients

import gin
import torch.nn as nn
from ptu.callbacks.callback_base import Callback


@gin.configurable
class GradClipping(Callback):
    def __init__(self, clip_value=1.0):
        # Value set in gin config
        self.clip_value = clip_value

    # Gradient gets clipped after backward pass
    def after_backward(self):
        nn.utils.clip_grad_norm_(self.trainer.model.parameters(), self.clip_value)
