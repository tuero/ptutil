# File: trainer.py
# Author: Jake Tuero (tuero@ualberta.ca)
# Date: April 26, 2021
#
# Trainer object which handles the main training logic,
# along with validation/testing
# Calls appropriate callbacks

import os
import sys
import torch
import gin
import gin.torch

from ptu.util.types import LoggingItem


@gin.configurable
class Trainer:
    def __init__(self, num_epochs, device, checkpoint_dir, cbs=[]):
        # From args/gin
        self.cbs = cbs
        for cb in self.cbs:
            cb.set_trainer(self)
        self.epoch_end = num_epochs
        self.device = torch.device(device)
        self.checkpoint_dir = checkpoint_dir
        # Bufferes
        self.logging_buffer = []
        self.metrics_buffer = []
        # Flags
        self.early_stoppage = False
        self.optimize = True
        self.step_global = 0
        self.in_train = False
        self.in_val = False
        self.in_test = False
        self.checkpoint_loaded = False

    # Calls appropriate callback
    def run_callbacks(self, callback_step):
        res = True
        for cb in self.cbs:
            res = res and getattr(cb, callback_step)()
        return res

    # Before the fit process
    def begin_fit(self):
        # Init optimizers, reset global count
        self.optimizer_infos = self.model.get_optimizers()
        self.step_global = 0
        self.early_stoppage = False
        self.model.begin_fit()
        return self.run_callbacks("begin_fit")

    # After model is trained
    def after_fit(self):
        self.model.after_fit()
        return self.run_callbacks("after_fit")

    # Before the validate process
    def begin_validate(self):
        self.model.begin_validate()
        return self.run_callbacks("begin_validate")
    
    # After the validate process
    def after_validate(self):
        self.model.after_validate()
        return self.run_callbacks("after_validate")

    # Before the test process
    def begin_test(self):
        self.model.begin_test()
        return self.run_callbacks("begin_test")

    # After the test process
    def after_test(self):
        self.model.after_test()
        return self.run_callbacks("after_test")

    # Before epoch start in training, validation, and testing
    def begin_epoch(self):
        self.model.begin_epoch()
        return self.run_callbacks("begin_epoch")

    # After the epoch in training, validation, and testing
    def after_epoch(self):
        self.model.after_epoch()
        return self.run_callbacks("after_epoch")

    # Before the epoch but only for train process
    def begin_train_step(self):
        self.step_global += 1
        self.in_train = True
        self.epoch_loss_train = 0.0
        self.model.begin_train_step()
        return self.run_callbacks("begin_train_step")

    # After the epoch but only for train process
    def after_train_step(self):
        # Step scheduler
        for optimizer_info in self.optimizer_infos:
            if optimizer_info.interval == "epoch":
                optimizer_info.scheduler.step()
        self.epoch_losses_train.append(self.epoch_loss_train)
        self.model.after_train_step()
        self.in_train = False
        return self.run_callbacks("after_train_step")

    # Before the epoch but only for validation process
    def begin_val_step(self):
        self.in_val = True
        self.epoch_loss_val = 0.0
        self.model.begin_val_step()
        return self.run_callbacks("begin_val_step")

    # After the epoch but only for validation process
    def after_val_step(self):
        self.epoch_losses_val.append(self.epoch_loss_val)
        self.model.after_val_step()
        self.in_val = False
        return self.run_callbacks("after_val_step")

    # Before the epoch but only for test process
    def begin_test_step(self):
        self.in_test = True
        self.model.begin_test_step()
        return self.run_callbacks("begin_test_step")

    # After the epoch but only for test process
    def after_test_step(self):
        self.model.after_test_step()
        self.in_test = False
        return self.run_callbacks("after_test_step")

    # Before every batch in training, validation, and testing
    def begin_batch(self):
        self.model.begin_batch()
        return self.run_callbacks("begin_batch")

    # After every batch in training, validation, and testing
    def after_batch(self):
        self.model.after_batch()
        return self.run_callbacks("after_batch")

    # After the loss is calculated
    def after_loss(self):
        self.model.after_loss()
        return self.run_callbacks("after_loss")

    # After the backward pass is done on the loss
    def after_backward(self):
        self.model.after_backward()
        return self.run_callbacks("after_backward")

    # -----------------------------------
    #      Trainer specific methods
    # -----------------------------------

    # Zeros the grad on each optimizer
    def zero_grad(self):
        for optimizer_info in self.optimizer_infos:
            optimizer_info.optimizer.zero_grad()

    # Steps the optimizers and scheduler
    def optimizer_step(self):
        # Update parameters with optimizers
        for optimizer_info in self.optimizer_infos:
            optimizer_info.optimizer.step()
        # Update any scheduler which has step interval
        for optimizer_info in self.optimizer_infos:
            if optimizer_info.interval == "step":
                optimizer_info.scheduler.step()

    # Single step specifically for training
    def train_step(self, batch, batch_idx):
        self.model.set_train()
        torch.set_grad_enabled(True)
        self.zero_grad()
        # Compute forward + loss
        self.loss = self.model.training_step(batch, batch_idx)
        self.epoch_loss_train += self.loss.item()
        self.after_loss()
        # Backward pass and update
        self.loss.backward()
        self.after_backward()
        self.optimizer_step()

    # Single step specifically for validation
    def val_step(self, batch, batch_idx):
        self.model.set_eval()
        torch.set_grad_enabled(False)
        self.loss = self.model.validation_step(batch, batch_idx)
        self.epoch_loss_val += self.loss.item()
        torch.set_grad_enabled(True)

    # Single step specifically for testing
    def test_step(self, batch, batch_idx):
        self.model.set_eval()
        torch.set_grad_enabled(False)
        self.model.testing_step(batch, batch_idx)
        torch.set_grad_enabled(True)

    # All steps necessary for training per epoch
    def train_loop(self, train_dataloader):
        self.begin_train_step()
        for batch_idx, batch in enumerate(train_dataloader):
            self.begin_batch()
            self.train_step(batch, batch_idx)
            self.after_batch()
        self.after_train_step()

    # All steps necessary for validation per epoch
    def val_loop(self, val_dataloader):
        self.begin_val_step()
        for batch_idx, batch in enumerate(val_dataloader):
            self.begin_batch()
            self.val_step(batch, batch_idx)
            self.after_batch()
        self.after_val_step()

    # All steps necessary for testing per epoch (only 1 epoch in testing)
    def test_loop(self, test_dataloader):
        self.begin_test_step()
        for batch_idx, batch in enumerate(test_dataloader):
            self.begin_batch()
            self.test_step(batch, batch_idx)
            self.after_batch()
        self.after_test_step()

    # The entire training process, includes validation
    def fit(self, model, train_dataloader=None, val_dataloader=None):
        self.model = model
        self.model.to(self.device)
        self.model.set_trainer(self)
        self.do_val = val_dataloader is not None
        self.num_batches_train = len(train_dataloader)
        self.num_batches_val = len(val_dataloader)

        # Checkpoint not loaded, start from 0 and dump config params
        if not self.checkpoint_loaded:
            self.epoch_start = 1
            self.logging_buffer.append(LoggingItem("INFO", gin.operative_config_str()))
            self.epoch_losses_train = []
            self.epoch_losses_val = []
        self.checkpoint_loaded = False

        # Train the model
        self.begin_fit()
        for epoch in range(self.epoch_start, self.epoch_end + 1):
            self.step_epoch = epoch
            self.begin_epoch()
            # Train loop
            self.train_loop(train_dataloader)
            # Validate loop
            if self.do_val:
                self.val_loop(val_dataloader)
            self.after_epoch()
            if self.early_stoppage:
                break
        self.after_fit()

    # The entire test process
    def test(self, model, test_dataloader):
        self.model = model
        self.model.to(self.device)
        self.model.set_trainer(self)
        self.size_test_data = len(test_dataloader)

        self.begin_test()
        self.test_loop(test_dataloader)
        self.after_test()

    # Saves the model/trainer as a checkpoint
    def save_checkpoint(self):
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
        checkpoint_file = os.path.join(self.checkpoint_dir, str(self.model) + ".pt")
        torch.save({**model_dict, **trainer_dict}, checkpoint_file)

    # Loads the trainer and model from checkpoint
    def load_from_checkpoint(self, model):
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
