import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.python.ops import math_ops


@tf.function
def SimpleRegressionAccuracy(y_true, y_pred, threshold=5):
    threshold = math_ops.cast(threshold, y_pred.dtype)
    y_true = math_ops.cast(y_true > threshold, y_true.dtype)
    y_pred = math_ops.cast(y_pred > threshold, y_pred.dtype)
    return math_ops.cast(math_ops.equal(y_true, y_pred), tf.keras.backend.floatx())

def discriminator_accuracy(fake_out, real_out):
    fake_acc = tf.keras.backend.mean(tf.keras.metrics.binary_accuracy(tf.zeros_like(fake_out), fake_out))
    real_acc = tf.keras.backend.mean(tf.keras.metrics.binary_accuracy(tf.ones_like(real_out), real_out))

    return fake_acc, real_acc

class CastingBinaryAccuracy(tf.keras.metrics.BinaryAccuracy):
    def __init__(self, name="Accuracy", dtype=None, threshold=0.5):
        super(CastingBinaryAccuracy, self).__init__(name=name, dtype=dtype, threshold=threshold)
        self.true_threshold = 5.0

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = math_ops.cast(y_true > self.true_threshold, y_true.dtype)
        super(CastingBinaryAccuracy, self).update_state(y_true, y_pred, sample_weight)


class MCC(tfa.metrics.MatthewsCorrelationCoefficient):
    def __init__(self, **kwargs):
        super(MCC, self).__init__(num_classes=1, name="MCC", **kwargs)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.expand_dims(tf.argmax(y_true, axis=-1), axis=-1)
        y_pred = tf.expand_dims(tf.argmax(y_pred, axis=-1), axis=-1)
        super(MCC, self).update_state(y_true, y_pred, sample_weight)

    def result(self):
        res = super(MCC, self).result()
        return res[0]


class WF1(tfa.metrics.F1Score):
    def __init__(self, num_classes=1, **kwargs):
        super(WF1, self).__init__(num_classes, average="weighted", name="WF1", **kwargs)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(tf.expand_dims(tf.argmax(y_true, axis=-1), axis=-1), dtype=tf.float32)
        y_pred = tf.cast(tf.expand_dims(tf.argmax(y_pred, axis=-1), axis=-1), dtype=tf.float32)
        super(WF1, self).update_state(y_true, y_pred, sample_weight)