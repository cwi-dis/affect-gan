import tensorflow as tf
from tensorflow.keras import callbacks
from tensorflow.python.eager import context
from tensorflow.python.ops import summary_ops_v2
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import itertools
import io
import os
from tensorboard.plugins.hparams import api as hp
from tensorflow.python.platform import tf_logging as logging
from util.misc import *
from data.inspector import plot_confusion_matrix, plot_to_image


class ConfusionMatrixCallback(callbacks.Callback):
    def __init__(self, val_data, logdir):
        super(ConfusionMatrixCallback, self).__init__()

        for data, label in val_data.take(1):
            self.val_data = data
            val_true_label = label

        self.val_true_labels = binary_discretize_labels(val_true_label)
        self.file_writer_cm = tf.summary.create_file_writer(os.path.join(logdir, 'image', 'cm'))

    def on_epoch_end(self, epoch, logs=None):
        eval_pred = self.model.predict(self.val_data)
        eval_pred = binary_discretize_labels(eval_pred, threshold=0.5)

        cm = confusion_matrix(self.val_true_labels, eval_pred)
        figure = plot_confusion_matrix(cm, class_names=["Low Arousal", "High Arousal"])
        cm_image = plot_to_image(figure)

        with self.file_writer_cm.as_default():
            tf.summary.image("Confusion Matrix", cm_image, step=epoch)
        super().on_epoch_end(epoch, logs)



class MetricsCallback(callbacks.TensorBoard):

    def __init__(self, logdir, *args, **kwargs):
        super(MetricsCallback, self).__init__(logdir, *args, **kwargs)
        self.best_eval = 0
        self.best_mcc = 0
        self.best_wf1 = 0

    def on_train_begin(self, logs=None):
        self.best_eval = 0
        super(MetricsCallback, self).on_train_begin(logs)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        current_acc = logs.get("val_accuracy")
        if current_acc is None:
            logging.warning('Best result tracking metric val_Accuracy '
                            'which is not available. Available metrics are: %s',
                            ','.join(list(logs.keys())))
        else:
            if np.greater(current_acc, self.best_eval):
                self.best_eval = current_acc
        logs.update({"val_best_acc": self.best_eval})

        current_mcc = logs.get("val_MCC")
        if current_mcc is None:
            logging.warning('Best result tracking metric val_MCC '
                            'which is not available. Available metrics are: %s',
                            ','.join(list(logs.keys())))
        else:
            if np.greater(current_mcc, self.best_mcc):
                self.best_mcc = current_mcc
        logs.update({"val_best_mcc": self.best_mcc})

        current_wf1 = logs.get("val_WF1")
        if current_wf1 is None:
            logging.warning('Best result tracking metric val_WF1 '
                            'which is not available. Available metrics are: %s',
                            ','.join(list(logs.keys())))
        else:
            if np.greater(current_wf1, self.best_wf1):
                self.best_wf1 = current_wf1
        logs.update({"val_best_wf1": self.best_wf1})

        super(MetricsCallback, self).on_epoch_end(epoch, logs)

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

class ThresholdEarlyStopping(callbacks.EarlyStopping):
    def __init__(self, min_epoch, *args, **kwargs):
        self.min_epoch = min_epoch
        super(ThresholdEarlyStopping, self).__init__(*args, **kwargs)

    def on_epoch_end(self, epoch, logs=None):
        current = self.get_monitor_value(logs)
        if current is None:
            return
        if self.monitor_op(current - self.min_delta, self.best):
            self.best = current
            self.wait = 0
            if self.restore_best_weights:
                self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if (self.wait >= self.patience) and (epoch >= self.min_epoch):
                self.stopped_epoch = epoch
                self.model.stop_training = True
                if self.restore_best_weights:
                    if self.verbose > 0:
                        print('Restoring model weights from the end of the best epoch.')
                    self.model.set_weights(self.best_weights)

class CallbacksProducer:

    def __init__(self, hparams, logdir, run_name, val_data=None):
        self.callbacks = {}
        self.logdir = logdir

        self.callbacks["base"] = MetricsCallback(
            logdir=self.logdir,
            update_freq="epoch"
        )

        #self.callbacks["lr_decay"] = callbacks.ReduceLROnPlateau(
        #    monitor='loss',
        #    factor=0.5,
        #    patience=4,
        #    min_lr=0.0001
        #)

        self.callbacks["early_stop"] = ThresholdEarlyStopping(
            min_epoch=20,
            monitor="val_MCC",
            mode="max",
            verbose=1,
            patience=6,
            restore_best_weights=True
        )

        #self.callbacks["early_stop"] = callbacks.EarlyStopping(
        #    monitor="val_loss",
        #    patience=15,
        #    mode="max",
        #    restore_best_weights=True
        #)

        #self.callbacks["confusion_matrix"] = ConfusionMatrixCallback(
        #    val_data=val_data,
        #    logdir=self.logdir
        #)

        if run_name is not None:
            self.callbacks["hparams"] = hp.KerasCallback(self.logdir, hparams, run_name)

    def get_callbacks(self, callback_ids=None):
        callback_list = []

        if callback_ids is None:
            for callback in self.callbacks.values():
                callback_list.append(callback)

        return callback_list
