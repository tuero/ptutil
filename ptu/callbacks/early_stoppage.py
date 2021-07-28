"""
File: early_stoppage.py
Author: Jake Tuero (tuero@ualberta.ca)
Date: April 26, 2021

Callback for saving checkpoints of the trainer/model
"""

import gin
from ptu.callbacks.callback_base import Callback
from ptu.util.types import LoggingItem


@gin.configurable
class EarlyStoppage(Callback):
    def __init__(self, patience: int):
        """Stops the learning process if no improvement after a set number of iterations.
        The metric to use for early stoppage should be manually stored in trainer.early_stoppage_buffer
        (maybe call in your model's after_epoch() method).

        Args:
            patience: The number of epochs of no improvement before learning stops
        """
        assert patience > 0
        self.patience = patience

    def _check_early_stoppage(self) -> None:
        if len(self.trainer.early_stoppage_buffer) == 0:
            return
        min_idx = self.trainer.early_stoppage_buffer.index(min(self.trainer.early_stoppage_buffer))
        losses_len = len(self.trainer.early_stoppage_buffer)
        if losses_len - min_idx > self.patience:
            self.trainer.early_stoppage = True
            self.trainer.logging_buffer.append(LoggingItem("INFO", "Breaking due to early stoppage."))

    def after_val_step(self) -> bool:
        """Early stoppage checked during end of validation"""
        self._check_early_stoppage()
        return True

    def after_episode(self) -> bool:
        """Early stoppage checked during end of an RL episode"""
        self._check_early_stoppage()
        return True
