import sys
import os
import logging
import gin
from ptu.callbacks.callback_base import Callback


# Callback for logging to both console and file
@gin.configurable
class Logger(Callback):
    def __init__(self, log_dir=None, experiment="", log_console=True):
        assert log_dir is not None or log_console, "Logger must have at least one of console or file options."

        # Set which loggers are active
        log_handlers = []
        if log_dir is not None:
            # Check if log directory exists
            log_path = os.path.join(log_dir, experiment + ".log")
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            log_handlers.append(logging.FileHandler(log_path))
        if log_console:
            log_handlers.append(logging.StreamHandler(sys.stdout))

        # Init loggers
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s] %(message)s",
            handlers=log_handlers,
        )

    # Logs from buffer are sent to which ever loggers active, and buffer cleared
    # Called at every callback point
    def log_out(self):
        for item in self.trainer.logging_buffer:
            logging.log(getattr(logging, item.level), item.msg)
        self.trainer.logging_buffer.clear()

    def begin_fit(self):
        self.log_out()
        return True

    def after_fit(self):
        self.log_out()
        return True

    def begin_validate(self):
        self.log_out()
        return True

    def after_validate(self):
        self.log_out()
        return True

    def begin_test(self):
        self.log_out()
        return True

    def after_test(self):
        self.log_out()
        return True

    def begin_epoch(self):
        self.log_out()
        return True

    def after_epoch(self):
        self.log_out()
        return True

    def begin_train_step(self):
        self.log_out()
        return True

    def after_train_step(self):
        self.log_out()
        return True

    def begin_val_step(self):
        self.log_out()
        return True

    def after_val_step(self):
        self.log_out()
        return True

    def begin_batch(self):
        self.log_out()
        return True

    def after_batch(self):
        self.log_out()
        return True

    def after_loss(self):
        self.log_out()
        return True

    def after_backward(self):
        self.log_out()
        return True
