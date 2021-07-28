"""
File: callback_base.py
Author: Jake Tuero (tuero@ualberta.ca)
Date: April 26, 2021

Base callback class
Methods called at respective times in trainer
"""
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ptu.base_trainer import BaseTrainer


class Callback:
    """Base callback object.
    Training objects will call callbacks at the respective time.
    """

    def set_trainer(self, trainer: BaseTrainer) -> None:
        """Stores a reference to the external trainer in case we need access to any buffers.
        """
        self.trainer = trainer

    def begin_fit(self) -> bool:
        """Called before the fit process."""
        return True

    def after_fit(self) -> bool:
        """Called after the model is trained."""
        return True

    def begin_test(self) -> bool:
        """Called before the test process."""
        return True

    def after_test(self) -> bool:
        """Called after the test process."""
        return True

    def begin_epoch(self) -> bool:
        """Called before the epoch start in training, validation, and testing."""
        return True

    def after_epoch(self) -> bool:
        """Called after the epoch in training, validation, and testing."""
        return True

    def begin_episode(self) -> bool:
        """Called before the episode start in rl training, similar to begin_epoch."""
        return True

    def after_episode(self) -> bool:
        """Called after the episode start in rl training, similar to begin_epoch."""
        return True

    def begin_train_step(self) -> bool:
        """Called before the epoch but only for train process."""
        return True

    def after_train_step(self) -> bool:
        """Called after the epoch but only for train process."""
        return True

    def begin_val_step(self) -> bool:
        """Called before the epoch but only for validation process."""
        return True

    def after_val_step(self) -> bool:
        """Called after the epoch but only for validation process."""
        return True

    def begin_test_step(self) -> bool:
        """Called before the epoch but only for test process."""
        return True

    def after_test_step(self) -> bool:
        """Called after the epoch but only for test process."""
        return True

    def begin_batch(self) -> bool:
        """Called before every batch in training, validation, and testing."""
        return True

    def after_batch(self) -> bool:
        """Called after every batch in training, validation, and testing."""
        return True

    def after_backward(self) -> bool:
        """Called after the backward pass is done on the loss."""
        return True
