"""
File: logger.py
Author: Jake Tuero (tuero@ualberta.ca)
Date: April 26, 2021

Callback for logging to both console and file
"""

import sys
import os
import logging
import gin
from ptu.callbacks.callback_base import Callback


@gin.configurable
class Logger(Callback):
    def __init__(self, log_dir: str = None, experiment: str = "", log_console: bool = True):
        """Logs various messages to the console/log file.
        To log a message, store a LoggingItem in the trainer.logging_buffer, and it will be pushed
        at the next opportunity. See types.LoggingItem for details.

        Args:
            log_dir: The base directory to store the logging files (optional)
            experiment: Experiment string name which will be used as a filename in log_dir (optional)
            log_console: Flag to log to console (optional)
        """
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
            format="%(asctime)s [%(levelname)-5.5s] %(message)s",
            handlers=log_handlers,
        )

    def log_out(self) -> None:
        """Logs from buffer are sent to which ever loggers active, and buffer cleared.
        Called at every callback point
        """
        for item in self.trainer.logging_buffer:
            logging.log(getattr(logging, item.level), item.msg)
        self.trainer.logging_buffer.clear()

    def begin_fit(self) -> bool:
        self.log_out()
        return True

    def after_fit(self) -> bool:
        self.log_out()
        return True

    def begin_test(self) -> bool:
        self.log_out()
        return True

    def after_test(self) -> bool:
        self.log_out()
        return True

    def begin_epoch(self) -> bool:
        self.log_out()
        return True

    def after_epoch(self) -> bool:
        self.log_out()
        return True

    def begin_episode(self) -> bool:
        self.log_out()
        return True

    def after_episode(self) -> bool:
        self.log_out()
        return True

    def begin_train_step(self) -> bool:
        self.log_out()
        return True

    def after_train_step(self) -> bool:
        self.log_out()
        return True

    def begin_val_step(self) -> bool:
        self.log_out()
        return True

    def after_val_step(self) -> bool:
        self.log_out()
        return True

    def begin_batch(self) -> bool:
        self.log_out()
        return True

    def after_batch(self) -> bool:
        self.log_out()
        return True

    def after_backward(self) -> bool:
        self.log_out()
        return True
