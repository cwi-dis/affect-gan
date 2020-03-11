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
from util.misc import *


def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image


def plot_confusion_matrix(cm, class_names):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.

    Args:
    cm (array, shape = [n, n]): a confusion matrix of integer classes
    class_names (array, shape = [n]): String names of the integer classes
    """
    figure = plt.figure(figsize=(2, 2))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Normalize the confusion matrix.
    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return figure


class ConfusionMatrixCallback(callbacks.Callback):
    def __init__(self, val_data, logdir):
        super(ConfusionMatrixCallback, self).__init__()

        for data, label in val_data.take(1):
            self.val_data = data
            val_true_label = label

        self.val_true_labels = binary_discretize_labels(val_true_label)
        self.file_writer_cm = tf.summary.create_file_writer(os.path.join(logdir, 'image', '/cm'))

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

    def __init__(self, hparams, logdir, run_name, val_data):
        self.callbacks = {}
        self.logdir = logdir

        self.callbacks["base"] = MetricsCallback(
            logdir=self.logdir
        )

        self.callbacks["lr_decay"] = callbacks.ReduceLROnPlateau(
            monitor='loss',
            factor=0.5,
            patience=4,
            min_lr=0.0001
        )

        self.callbacks["early_stop"] = callbacks.EarlyStopping(
            monitor="val_loss",
            patience=8
        )

        self.callbacks["confusion_matrix"] = ConfusionMatrixCallback(
            val_data=val_data,
            logdir=self.logdir
        )

        if run_name is not None:
            self.callbacks["hparams"] = hp.KerasCallback(self.logdir, hparams, run_name)

    def get_callbacks(self, callback_ids=None):
        callback_list = []

        if callback_ids is None:
            for callback in self.callbacks.values():
                callback_list.append(callback)

        return callback_list
