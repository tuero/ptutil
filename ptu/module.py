# File: module.py
# Author: Jake Tuero (tuero@ualberta.ca)
# Date: April 26, 2021
#
# Module object which contains the model,
# interfaces with trainer and callbacks
# @NOTE: init should initialize the optimizer_infos

import torch.nn as nn


class PTUtilModule(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.optimizer_infos = None

    # Saves a trainer reference so we can pull trainer info
    def set_trainer(self, trainer):
        self.trainer = trainer

    # Returns references to the optimizers
    def get_optimizers(self):
        assert self.optimizer_infos is not None
        return self.optimizer_infos

    # String name for checkpoints
    def __str__(self):
        raise NotImplementedError

    # Returns model output
    def forward(self, X):
        raise NotImplementedError

    # Single step during training phase
    def training_step(self, batch, batch_idx):
        # compute forward
        # calc loss
        raise NotImplementedError

    # Single step during validation phase
    def validation_step(self, batch, batch_idx):
        raise NotImplementedError

    # Single step during testing phase
    def testing_step(self, batch, batch_idx):
        raise NotImplementedError

    # Gets necessary info for checkpoint
    def get_checkpoint(self):
        model_dict = {"model": self.state_dict()}
        for i, optimizer_info in enumerate(self.optimizer_infos):
            model_dict["optimizer_{}".format(i)] = optimizer_info.optimizer.state_dict()
            model_dict["scheduler_{}".format(i)] = optimizer_info.scheduler.state_dict()
        return model_dict

    # Loads model from checkpoint
    def load_from_checkpoint(self, checkpoint):
        self.load_state_dict(checkpoint["model"])
        for i, optimizer_info in enumerate(self.optimizer_infos):
            optimizer_info.optimizer.load_state_dict(checkpoint["optimizer_{}".format(i)])
            optimizer_info.scheduler.load_state_dict(checkpoint["optimizer_{}".format(i)])

    # Put model into train mode
    def set_train(self):
        self.train(True)

    # Put model into test mode
    def set_eval(self):
        self.train(False)

    # -----------------------------------
    # Callback matching steps for trainer
    # -----------------------------------

    # Before the fit process
    def begin_fit(self):
        return

    # After model is trained
    def after_fit(self):
        return

    # Before the validate process
    def begin_validate(self):
        return

    # After the validate process
    def after_validate(self):
        return

    # Before the test process
    def begin_test(self):
        return

    # After the test process
    def after_test(self):
        return

    # Before epoch start in training, validation, and testing
    def begin_epoch(self):
        return

    # After the epoch in training, validation, and testing
    def after_epoch(self):
        return

    # Before the epoch but only for train process
    def begin_train_step(self):
        self.loss_train = 0

    # After the epoch but only for train process
    def after_train_step(self):
        return

    # Before the epoch but only for validation process
    def begin_val_step(self):
        self.loss_val = 0

    # After the epoch but only for validation process
    def after_val_step(self):
        return

    # Before the epoch but only for test process
    def begin_test_step(self):
        return

    # After the epoch but only for test process
    def after_test_step(self):
        return

    # Before every batch in training, validation, and testing
    def begin_batch(self):
        return

    # After every batch in training, validation, and testing
    def after_batch(self):
        return

    # After the loss is calculated
    def after_loss(self):
        return

    # After the backward pass is done on the loss
    def after_backward(self):
        return
