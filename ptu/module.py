"""
File: module.py
Author: Jake Tuero (tuero@ualberta.ca)
Date: April 26, 2021

Module object which contains the model,
interfaces with trainer and callbacks
"""
from __future__ import annotations

import torch
import torch.nn as nn
from ptu.callbacks.grad_clipping import GradClipping

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ptu.trainer import Trainer
    from typing import Dict, Any


class PTUtilModule(nn.Module):
    def __init__(self, *args, **kwargs):
        """Module object, which holds the necessary models and is called by the trainer.
        This extends nn.Module, so that the entire sub-models can be treated as one large one.
        NOTE: Initializer should initialize the optimizers inside self.optimizer_infos,
        so that the optimizer states can be saved/loaded for checkpointing.
        """
        super().__init__(*args, **kwargs)
        self.optimizer_infos = None

    def set_trainer(self, trainer: Trainer) -> None:
        """Saves a trainer reference so we can pull trainer info"""
        self.trainer = trainer

    def optimization_step(
        self, network: torch.nn.Module, optimizer: torch.optim.Optimizer, loss: torch.tensor
    ) -> float:
        """Perform an optimization step of the given network with respect to the given loss.
        This should be manually called in self.training_step().
        If gradient clipping callback is set on the trainer, then it will get called here on the given model.

        Args:
            network: The network to update (needed for any callbacks which need the networks parameters)
            optimizer: The optimizer whose params should match the given network's params
            loss: Loss for updating the network

        Returns:
            The loss value.
        """
        optimizer.zero_grad()
        self.trainer.loss = loss
        _loss = loss.item()
        self.trainer.grad_clip_buffer.append(network)
        if self.trainer.use_amp:
            self.trainer.scaler.scale(loss).backward()
            if True in [isinstance(c, GradClipping) for c in self.trainer.cbs]:
                self.trainer.scaler.unscale_(optimizer)
            self.trainer._after_backward()  # Will run grad_clipping if necessary
            self.trainer.scaler.step(optimizer)
            self.trainer.scaler.update()
        else:
            loss.backward()
            self.trainer._after_backward()  # Will run grad_clipping if necessary
            optimizer.step()
        return _loss

    def __str__(self):
        """String name for checkpoints."""
        raise NotImplementedError

    def forward(self, X):
        """Runs the model output on a given sample. This isn't necessary to implement.
        An example when it may be implemented is if the trained model wants to be called for inference later on.
        """
        raise NotImplementedError

    def training_step(self, batch: torch.tensor, batch_idx: int) -> float:
        """Runs a single step during the training phase for a single batch.
        The loss should be calculated here, with self.optimization_step called to update model parameters.

        Args:
            batch: The current batch
            batch_idx: The index of the batch

        Returns:
            The loss value for the given batch
        """
        raise NotImplementedError

    def validation_step(self, batch: torch.tensor, batch_idx: int) -> float:
        """Runs a single step during the validation phase for a single batch.

        Args:
            batch: The current batch
            batch_idx: The index of the batch

        Returns:
            The loss value for the given batch
        """
        raise NotImplementedError

    def testing_step(self, batch: torch.tensor, batch_idx: int):
        """Runs a single step during the test phase for a single batch.
        This can be very general, as testing may involve calculating various metrics.

        Args:
            batch: The current batch
            batch_idx: The index of the batch
        """
        raise NotImplementedError

    def get_checkpoint(self) -> Dict[str, Any]:
        """Creats a checkpoint dictionary for the trainer object.
        If this method is overwritten, the optimizers in self.optimizer_info needs to be manually stored.
        This is called during trainer.save_checkpoint().

        Returns:
            Dictionary holding stored state info
        """
        model_dict = {"model": self.state_dict()}
        for i, optimizer_info in enumerate(self.optimizer_infos):
            model_dict["optimizer_{}".format(i)] = optimizer_info.optimizer.state_dict()
            model_dict["scheduler_{}".format(i)] = optimizer_info.scheduler.state_dict()
        return model_dict

    def load_from_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Sets the model state from the given checkpoint.
        If self.get_checkpoint() is overwritten, this must also be overwritten to match.

        Args:
            checkpoint: Checkpoint given from the trainer object.
        """
        self.load_state_dict(checkpoint["model"])
        for i, optimizer_info in enumerate(self.optimizer_infos):
            optimizer_info.optimizer.load_state_dict(checkpoint["optimizer_{}".format(i)])
            optimizer_info.scheduler.load_state_dict(checkpoint["optimizer_{}".format(i)])

    def set_train(self) -> None:
        """Puts the model into train mode."""
        self.train(True)

    def set_eval(self) -> None:
        """Puts the model into eval mode."""
        self.train(False)

    # -----------------------------------
    # Callback matching steps for trainer
    # -----------------------------------

    def begin_fit(self) -> None:
        """Called before the fit process."""
        return

    def after_fit(self) -> None:
        """Called after the model is trained."""
        return

    def begin_test(self) -> None:
        """Called before the test process."""
        return

    def after_test(self) -> None:
        """Called after the test process."""
        return

    def begin_epoch(self) -> None:
        """Called before the epoch start in training, validation, and testing."""
        return

    def after_epoch(self) -> None:
        """Called after the epoch in training, validation, and testing."""
        return

    def begin_train_step(self) -> None:
        """Called before the epoch but only for train process."""
        self.loss_train = 0

    def after_train_step(self):
        """Called after the epoch but only for train process."""
        return

    def begin_val_step(self) -> None:
        """Called before the epoch but only for validation process."""
        self.loss_val = 0

    def after_val_step(self) -> None:
        """Called after the epoch but only for validation process."""
        return

    def begin_test_step(self) -> None:
        """Called before the epoch but only for test process."""
        return

    def after_test_step(self) -> None:
        """Called after the epoch but only for test process."""
        return

    def begin_batch(self) -> None:
        """Called before every batch in training, validation, and testing."""
        return

    def after_batch(self) -> None:
        """Called after every batch in training, validation, and testing."""
        return

    def after_backward(self) -> None:
        """Called after the backward pass is done on the loss."""
        return
