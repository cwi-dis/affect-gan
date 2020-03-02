import tensorflow as tf


@tf.function
def SimpleRegressionAccuracy(y_true, y_pred):
    y_true = tf.map_fn(lambda x: 0.0 if x < 5 else 1.0, y_true)
    y_pred = tf.map_fn(lambda x: 0.0 if x < 5 else 1.0, y_pred)
    return tf.metrics.binary_accuracy(y_true, y_pred)