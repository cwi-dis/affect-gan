import tensorflow as tf
from tensorflow.python.ops import math_ops


@tf.function
def SimpleRegressionAccuracy(y_true, y_pred, threshold=5):
    threshold = math_ops.cast(threshold, y_pred.dtype)
    y_true = math_ops.cast(y_true > threshold, y_true.dtype)
    y_pred = math_ops.cast(y_pred > threshold, y_pred.dtype)
    return math_ops.cast(math_ops.equal(y_true, y_pred), tf.keras.backend.floatx())


class CastingBinaryAccuracy(tf.keras.metrics.BinaryAccuracy):
    def __init__(self, name="Accuracy", dtype=None, threshold=0.5):
        super(CastingBinaryAccuracy, self).__init__(name=name, dtype=dtype, threshold=threshold)
        self.true_threshold = 5.0

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = math_ops.cast(y_true > self.true_threshold, y_true.dtype)
        super(CastingBinaryAccuracy, self).update_state(y_true, y_pred, sample_weight)
