import tensorflow as tf
from tensorflow.keras import callbacks
from tensorflow.python.eager import context
from tensorflow.python.ops import summary_ops_v2

from datetime import datetime
import os


class MetricsCallback(callbacks.TensorBoard):

    def __init__(self, logdir, *args, **kwargs):
        super(MetricsCallback, self).__init__(logdir, *args, **kwargs)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        lr = self.model.optimizer.lr
        logs.update({'lr': tf.keras.backend.eval(lr)})
        super().on_epoch_end(epoch, logs)

    def _log_metrics(self, logs, prefix, step):
        """Writes metrics out as custom scalar summaries.
        Arguments:
            logs: Dict. Keys are scalar summary names, values are NumPy scalars.
            prefix: String. The prefix to apply to the scalar summary names.
            step: Int. The global step to use for TensorBoard.
        """
        if logs is None:
            logs = {}

        # Group metrics by the name of their associated file writer. Values
        # are lists of metrics, as (name, scalar_value) pairs.
        logs_by_writer = {
            self._train_run_name: [],
            self._validation_run_name: [],
        }
        validation_prefix = 'val_'
        for (name, value) in logs.items():
            if name in ('batch', 'size', 'num_steps'):
                # Scrub non-metric items.
                continue
            if name.startswith(validation_prefix):
                name = name[len(validation_prefix):]
                writer_name = self._validation_run_name
            else:
                writer_name = self._train_run_name
            name = prefix + name  # assign batch or epoch prefix
            logs_by_writer[writer_name].append((name, value))

        with context.eager_mode():
            with summary_ops_v2.always_record_summaries():
                for writer_name in logs_by_writer:
                    these_logs = logs_by_writer[writer_name]
                    if not these_logs:
                        # Don't create a "validation" events file if we don't
                        # actually have any validation data.
                        continue
                    writer = self._get_writer(writer_name)
                    with writer.as_default():
                        for (name, value) in these_logs:
                            summary_ops_v2.scalar(name, value, step=step)


class CallbacksProducer:

    def __init__(self, logdir="../Logs"):
        self.callbacks = {}
        self.logdir = os.path.join(logdir,datetime.now().strftime("%Y%m%d-%H%M%S"))

        self.callbacks["base"] = MetricsCallback(
            logdir=self.logdir
        )

        self.callbacks["lr_decay"] = callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=7,
            min_lr=0.0001
        )

        #self.callbacks["early_stop"] = callbacks.EarlyStopping(
        #    monitor="val_loss",
        #    patience=10
        #)

    def get_callbacks(self, callback_ids=None):
        callback_list = []

        if callback_ids is None:
            for callback in self.callbacks.values():
                callback_list.append(callback)

        return callback_list
