"""
File: trainer.py
Author: Jake Tuero (tuero@ualberta.ca)
Date: April 26, 2021

Trainer object which handles the main training logic,
along with validation/testing
Calls appropriate callbacks
"""
from __future__ import annotations
from ptu.base_trainer import BaseTrainer

import os
import sys
import torch
from torch.utils.data.dataloader import DataLoader
import gin
import gin.torch

from ptu.module import PTUtilModule
from ptu.util.types import LoggingItem

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ptu.callbacks.callback_base import Callback
    from typing import List


@gin.configurable
class Trainer(BaseTrainer):
    def __init__(
        self,
        num_epochs: int,
        device: torch.device,
        checkpoint_dir: str,
        cbs: List[Callback] = [],
        use_amp: bool = False,
    ):
        """Handles the training, validation, and testing process for a PTUtilModule.
        Passed arguments can be set by a gin config.
        NOTE: Use AMP at discretion, as it is not fully tested and there may be issues when
        combining with gradient clipping.

        Args:
            num_epochs: Maximum number of epochs to train for
            device: device for which all objects will reference
            checkpoint_dir: relative path (from script run) for checkpoint saving
            cbs: callbacks for the trainer to use
            use_amp: flag for whether to use automatic mixed precision
        """
        super().__init__(device, checkpoint_dir, cbs, use_amp)
        self.epoch_end = num_epochs
        # Flags
        self.in_train = False
        self.in_val = False
        self.in_test = False
        self.step_epoch = 0
        self.step_global = 0
        self.epoch_losses_train = []
        self.epoch_losses_val = []

    def _begin_fit(self) -> None:
        """Called before the fit process"""
        # Init optimizers, reset global count
        self.early_stoppage = False
        self.model.begin_fit()
        return self._run_callbacks("begin_fit")

    def _after_fit(self) -> None:
        """Called after the fit process (model is fully trained)"""
        self.model.after_fit()
        return self._run_callbacks("after_fit")

    def _begin_test(self) -> None:
        """Called before the test process"""
        self.model.begin_test()
        return self._run_callbacks("begin_test")

    def _after_test(self) -> None:
        """Called after the test process"""
        self.model.after_test()
        return self._run_callbacks("after_test")

    def _begin_epoch(self) -> None:
        """Called before epoch start in training, validation, and testing"""
        self.model.begin_epoch()
        return self._run_callbacks("begin_epoch")

    def _after_epoch(self) -> None:
        """Called after epoch start in training, validation, and testing"""
        self.model.after_epoch()
        return self._run_callbacks("after_epoch")

    def _begin_train_step(self) -> None:
        """Called before the epoch but only for train process"""
        self.in_train = True
        self.epoch_loss_train = 0.0
        self.model.begin_train_step()
        return self._run_callbacks("begin_train_step")

    def _after_train_step(self) -> None:
        """Called after the epoch but only for train process"""
        self.epoch_losses_train.append(self.epoch_loss_train)
        self.model.after_train_step()
        self.in_train = False
        return self._run_callbacks("after_train_step")

    def _begin_val_step(self) -> None:
        """Called before the epoch but only for validation process"""
        self.in_val = True
        self.epoch_loss_val = 0.0
        self.model.begin_val_step()
        return self._run_callbacks("begin_val_step")

    def _after_val_step(self) -> None:
        """Called after the epoch but only for validation process"""
        self.epoch_losses_val.append(self.epoch_loss_val)
        self.model.after_val_step()
        self.in_val = False
        return self._run_callbacks("after_val_step")

    def _begin_test_step(self) -> None:
        """Called before the epoch but only for test process"""
        self.in_test = True
        self.model.begin_test_step()
        return self._run_callbacks("begin_test_step")

    def _after_test_step(self) -> None:
        """Called after the epoch but only for test process"""
        self.model.after_test_step()
        self.in_test = False
        return self._run_callbacks("after_test_step")

    def _begin_batch(self) -> None:
        """Called before every batch in training, validation, and testing"""
        self.model.begin_batch()
        return self._run_callbacks("begin_batch")

    def _after_batch(self) -> None:
        """Called after every batch in training, validation, and testing"""
        self.model.after_batch()
        return self._run_callbacks("after_batch")

    def _after_backward(self) -> None:
        """Called after the backward pass is done on the loss.
        This is generally called after model.optimization_step().
        Gradient clipping will be called if the callback is set.
        """
        self.grad_clip_buffer.append(self.model)
        self.model.after_backward()
        return self._run_callbacks("after_backward")

    # -----------------------------------
    #      Trainer specific methods
    # -----------------------------------

    def _train_step(self, batch: torch.tensor, batch_idx: int) -> None:
        """Runs a single step specifically for training"""
        self.step_global += 1
        self.model.set_train()
        torch.set_grad_enabled(True)
        with torch.cuda.amp.autocast(self.use_amp):
            self.loss = self.model.training_step(batch, batch_idx)
        self.epoch_loss_train += self.loss

    def _val_step(self, batch: torch.tensor, batch_idx: int) -> None:
        """Runs a single step specifically for validation"""
        self.model.set_eval()
        torch.set_grad_enabled(False)
        with torch.cuda.amp.autocast(self.use_amp):
            self.loss = self.model.validation_step(batch, batch_idx)
        self.epoch_loss_val += self.loss
        torch.set_grad_enabled(True)

    def _test_step(self, batch: torch.tensor, batch_idx: int) -> None:
        """Runs a single step specifically for testing"""
        self.model.set_eval()
        torch.set_grad_enabled(False)
        with torch.cuda.amp.autocast(self.use_amp):
            self.model.testing_step(batch, batch_idx)
        torch.set_grad_enabled(True)

    def _train_loop(self, train_dataloader: DataLoader) -> None:
        """Runs all steps necessary for training per epoch"""
        self._begin_train_step()
        for batch_idx, batch in enumerate(train_dataloader):
            self._begin_batch()
            self._train_step(batch, batch_idx)
            self._after_batch()
        self._after_train_step()

    def _val_loop(self, val_dataloader: DataLoader) -> None:
        """Runs all steps necessary for validation per epoch"""
        self._begin_val_step()
        for batch_idx, batch in enumerate(val_dataloader):
            self._begin_batch()
            self._val_step(batch, batch_idx)
            self._after_batch()
        self._after_val_step()

    def _test_loop(self, test_dataloader: DataLoader) -> None:
        """ll steps necessary for testing per epoch (only 1 epoch in testing)"""
        self._begin_test_step()
        for batch_idx, batch in enumerate(test_dataloader):
            self._begin_batch()
            self._test_step(batch, batch_idx)
            self._after_batch()
        self._after_test_step()

    # -----------------------------------
    #      Public Methods
    # -----------------------------------

    def fit(self, model: PTUtilModule, train_dataloader: DataLoader, val_dataloader: DataLoader = None) -> None:
        """Fits the given model, calling the training API calls.

        Args:
            model: The model to train
            train_dataloader: The dataloader containing the training data
            train_dataloader: The dataloader containing the validation data (optional)
        """
        assert isinstance(model, PTUtilModule), "Model needs to be of derived type PTUtilModule"
        self.model = model
        self.model.to(self.device)
        self.model.set_trainer(self)
        self.do_val = val_dataloader is not None
        self.num_batches_train = len(train_dataloader)
        self.num_batches_val = len(val_dataloader) if val_dataloader is not None else -1

        # Checkpoint not loaded, start from 0 and dump config params
        if not self.checkpoint_loaded:
            self.epoch_start = 1
            self.step_global = 0
            self.logging_buffer.append(LoggingItem("INFO", gin.operative_config_str()))
            self.epoch_losses_train = []
            self.epoch_losses_val = []
        self.checkpoint_loaded = False

        # Train the model
        self._begin_fit()
        for epoch in range(self.epoch_start, self.epoch_end + 1):
            self.step_epoch = epoch
            self._begin_epoch()
            # Train loop
            self._train_loop(train_dataloader)
            # Validate loop
            if self.do_val:
                self._val_loop(val_dataloader)
            self._after_epoch()
            if self.early_stoppage:
                break
        self._after_fit()

    def test(self, model: PTUtilModule, test_dataloader: DataLoader) -> None:
        """Tests the trained model on the given dataloader.

        Args:
            model: The model to train
            train_dataloader: The dataloader containing the test data
        """
        assert isinstance(model, PTUtilModule), "Model needs to be of derived type PTUtilModule"
        self.model = model
        self.model.to(self.device)
        self.model.set_trainer(self)
        self.size_test_data = len(test_dataloader)

        self._begin_test()
        self._test_loop(test_dataloader)
        self._after_test()

    def save_checkpoint(self, str_model: str = None) -> None:
        """Saves the trainer state along with the current model.
        This is automatically called when using the checkpoint callback.
        """
        model_dict = self.model.get_checkpoint()
        trainer_dict = {
            "epoch": self.step_epoch + 1,
            "step_global": self.step_global,
            "epoch_losses_train": self.epoch_losses_train,
            "epoch_losses_val": self.epoch_losses_val,
        }
        # check if directory exists, create if not
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        str_model = str_model if str_model is not None else str(self.model)
        checkpoint_file = os.path.join(self.checkpoint_dir, str_model + ".pt")
        torch.save({**model_dict, **trainer_dict}, checkpoint_file)

    def load_from_checkpoint(self, model: PTUtilModule) -> None:
        """Loads the trainer and model from checkpoint.
        This should be called externally from the main module.

        Args:
            model: Reference to an untrained model to load into
        """
        # Load model/checkpoint to device
        model.to(self.device)
        checkpoint_file = os.path.join(self.checkpoint_dir, str(model) + ".pt")
        try:
            checkpoint = torch.load(checkpoint_file, map_location=self.device)
            # Set internal variables from checkpoint
            self.epoch_start = checkpoint["epoch"]
            self.step_global = checkpoint["step_global"]
            self.epoch_losses_train = checkpoint["epoch_losses_train"]
            self.epoch_losses_val = checkpoint["epoch_losses_val"]
            model.load_from_checkpoint(checkpoint)
            # Set flags and cleanup
            self.checkpoint_loaded = True
            self.logging_buffer.append(LoggingItem("INFO", "Loading from checkpoint."))
            del checkpoint
        except:
            sys.exit("Error: Unable to open config file {}".format(checkpoint_file))
