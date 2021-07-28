"""
File: checkpoint.py
Author: Jake Tuero (tuero@ualberta.ca)
Date: April 26, 2021

Callback for saving checkpoints of the trainer/model
"""
from ptu.callbacks.callback_base import Callback


class Checkpoint(Callback):
    def __init__(self):
        """Calls the trainer save_checkpoint method after each epoch/episode,
        as well as once the training process is complete.
        """
        pass

    def after_fit(self) -> bool:
        """Final checkpoint saved after training process is complete"""
        self.trainer.save_checkpoint()
        return True

    def after_epoch(self) -> bool:
        """Checkpoint saved after every epoch complete (training & validation)"""
        self.trainer.save_checkpoint()
        return True

    def after_episode(self) -> bool:
        """Checkpoint saved after every episode completes"""
        if self.trainer.save_checkpoint_flag:
            self.trainer.save_checkpoint()
            self.trainer.save_checkpoint_flag = False
        return True
