# File: early_stoppage.py
# Author: Jake Tuero (tuero@ualberta.ca)
# Date: April 26, 2021
#
# Callback for saving checkpoints of the trainer/model

import gin
from ptu.callbacks.callback_base import Callback
from ptu.util.types import LoggingItem


@gin.configurable
class EarlyStoppage(Callback):
    def __init__(self, patience):
        assert patience > 0
        self.patience = patience

    # Early stoppage checked during end of validation
    def after_val_step(self):
        min_idx = self.trainer.epoch_losses_val.index(min(self.trainer.epoch_losses_val))
        losses_len = len(self.trainer.epoch_losses_val)
        if losses_len - min_idx > self.patience:
            self.trainer.early_stoppage = True
            self.trainer.logging_buffer.append(LoggingItem("INFO", "Breaking due to early stoppage."))
        return True
