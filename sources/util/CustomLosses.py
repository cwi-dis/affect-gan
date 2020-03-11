import tensorflow as tf
from tensorflow.keras import losses
from tensorflow.python.ops import math_ops


class CastingBinaryCrossentropy(losses.BinaryCrossentropy):
    def __init__(self, **kwargs):
        super(CastingBinaryCrossentropy).__init__(**kwargs)
        self.threshold = 5.0

    def call(self, y_true, y_pred):
        y_true = math_ops.cast(y_true > self.threshold, y_true.dtype)
        return super(CastingBinaryCrossentropy, self).call(y_true, y_pred)
