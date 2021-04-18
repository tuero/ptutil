# Base callback class
# Methods called at respective times in trainer
class Callback:
    def set_trainer(self, trainer):
        self.trainer = trainer
    
    # Before the fit process
    def begin_fit(self):
        return True

    # After model is trained
    def after_fit(self):
        return True

    # Before the validate process
    def begin_validate(self):
        return True

    # After the validate process
    def after_validate(self):
        return True

    # Before the test process
    def begin_test(self):
        return True

    # After the test process
    def after_test(self):
        return True

    # Before epoch start in training, validation, and testing
    def begin_epoch(self):
        return True

    # After the epoch in training, validation, and testing
    def after_epoch(self):
        return True

    # Before the epoch but only for train process
    def begin_train_step(self):
        return True

    # After the epoch but only for train process
    def after_train_step(self):
        return True

    # Before the epoch but only for validation process
    def begin_val_step(self):
        return True

    # After the epoch but only for validation process
    def after_val_step(self):
        return True

    # Before the epoch but only for test process
    def begin_test_step(self):
        return True

    # After the epoch but only for test process
    def after_test_step(self):
        return True

    # Before every batch in training, validation, and testing
    def begin_batch(self):
        return True

    # After every batch in training, validation, and testing
    def after_batch(self):
        return True

    # After the loss is calculated
    def after_loss(self):
        return True

    # After the backward pass is done on the loss
    def after_backward(self):
        return True
