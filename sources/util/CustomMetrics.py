import tensorflow as tf
from tensorflow.python.ops import math_ops


@tf.function
def SimpleRegressionAccuracy(y_true, y_pred, threshold=5):
    threshold = math_ops.cast(threshold, y_pred.dtype)
    y_true = math_ops.cast(y_true > threshold, y_true.dtype)
    y_pred = math_ops.cast(y_pred > threshold, y_pred.dtype)
    return math_ops.cast(math_ops.equal(y_true, y_pred), tf.keras.backend.floatx())