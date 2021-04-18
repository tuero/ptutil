from ptu.callbacks.callback_base import Callback


# Callback for saving checkpoints of the trainer/model
class Checkpoint(Callback):
    # Final checkpoint saved after training process is complete
    def after_fit(self):
        self.trainer.save_checkpoint()
        return True

    # Checkpoint saved after every epoch complete (training & validation)
    def after_epoch(self):
        self.trainer.save_checkpoint()
        return True
