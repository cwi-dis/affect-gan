import tensorflow as tf
from tensorflow.keras import callbacks
from datetime import datetime


class LRLogCallback(callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        lr = self.model.optimizer.lr
        logs.update({'lr': lr})
        super().on_epoch_end(epoch, logs)


class CallbacksProducer:

    def __init__(self, logdir="../../Logs/Baseline/"):
        self.callbacks = {}
        self.logdir = logdir + datetime.now().strftime("%Y%m%d-%H%M%S")

        self.callbacks["base"] = callbacks.TensorBoard(
            log_dir=logdir,
            histogram_freq=1,
            write_graph=False,
            write_images=True,
            update_freq=100
        )

        self.callbacks["lr_decay"] = callbacks.ReduceLROnPlateau(
            monitor='train_loss',
            factor=0.75,
            patience=5,
            min_lr=0.0001
        )

        #self.callbacks["early_stop"] = callbacks.EarlyStopping(
        #    monitor="val_loss",
        #    patience=10
        #)

        self.callbacks["lr_monitor"] = LRLogCallback()

    def get_callbacks(self, callback_ids=None):
        callback_list = []

        if callback_ids is None:
            for callback in self.callbacks.values():
                callback_list.append(callback)

        return callback_list
