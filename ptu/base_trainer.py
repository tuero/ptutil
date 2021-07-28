"""
File: base_trainer.py
Author: Jake Tuero (tuero@ualberta.ca)
Date: April 26, 2021

Base trainer for is used by the standard/RL trainer.
"""
from __future__ import annotations

import torch
import gin
import gin.torch

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ptu.callbacks.callback_base import Callback
    from typing import List


@gin.configurable
class BaseTrainer:
    def __init__(self, device: torch.device, checkpoint_dir: str, cbs: List[Callback] = [], use_amp: bool = True):
        """Base class for training objects

        Args:
            device: device for which all objects will reference
            checkpoint_dir: relative path (from script run) for checkpoint saving
            cbs: callbacks for the trainer to use
            use_amp: flag for whether to use automatic mixed precision
        """
        # From args/gin
        self.cbs = cbs
        for cb in self.cbs:
            cb.set_trainer(self)
        self.device = torch.device(device)
        self.checkpoint_dir = checkpoint_dir
        # Bufferes
        self.logging_buffer = []
        self.metrics_buffer = []
        self.grad_clip_buffer = []
        self.early_stoppage_buffer = []
        # Flags
        self.save_checkpoint_flag = False
        self.early_stoppage = False
        self.step_global = 0
        self.checkpoint_loaded = False
        self.scaler = torch.cuda.amp.GradScaler()
        self.use_amp = bool(use_amp)

    def _run_callbacks(self, callback_step: str):
        """Calls the appropriate callbacks for the given step"""
        res = True
        for cb in self.cbs:
            res = res and getattr(cb, callback_step)()
        return res

    def get_num_params(self, model: torch.nn.Module) -> int:
        """Get the number of trainable parameters for the given model.

        Args:
            model: The model to paramter count.

        Returns:
            The count of trainable parameters for the model
        """
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def save_checkpoint(self):
        raise NotImplementedError

    def load_from_checkpoint(self, model):
        raise NotImplementedError
