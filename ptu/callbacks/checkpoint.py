# File: checkpoint.py
# Author: Jake Tuero (tuero@ualberta.ca)
# Date: April 26, 2021
#
# Callback for saving checkpoints of the trainer/model

from ptu.callbacks.callback_base import Callback


class Checkpoint(Callback):
    # Final checkpoint saved after training process is complete
    def after_fit(self):
        self.trainer.save_checkpoint()
        return True

    # Checkpoint saved after every epoch complete (training & validation)
    def after_epoch(self):
        self.trainer.save_checkpoint()
        return True
