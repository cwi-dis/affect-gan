import tensorflow as tf
from tensorflow.keras import losses

class CombinedLoss(losses.Loss):
    def call(self, y_true, y_pred):
        return losses.MSE(y_true, y_pred)