import tensorflow as tf
from tensorflow.keras import losses
from tensorflow.python.ops import math_ops

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)

class CastingBinaryCrossentropy(losses.BinaryCrossentropy):
    def __init__(self, **kwargs):
        super(CastingBinaryCrossentropy, self).__init__(**kwargs)
        self.threshold = 5.0

    def call(self, y_true, y_pred):
        y_true = math_ops.cast(y_true > self.threshold, y_true.dtype)
        return super(CastingBinaryCrossentropy, self).call(y_true, y_pred)


def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def wgangp_critic_loss(real_out, fake_out, interpolated_out, alpha=10):
    gradient_penalty = alpha * tf.square(tf.subtract(tf.norm(interpolated_out, ord="euclidean", axis=1), 1))
    wgangp_loss = fake_out - real_out + gradient_penalty

    return tf.reduce_mean(wgangp_loss)
