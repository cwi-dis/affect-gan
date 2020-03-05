import tensorflow as tf
import tensorflow.keras.layers as layers

import config

class AttentionNET(tf.keras.Model):

    def __init__(self, hparams, filters=5):
        super(AttentionNET, self).__init__()

        self.query_mat = layers.Conv1D(
            filters=filters,
            kernel_size=4,
            strides=2
        )
        self.value_mat = layers.Conv1D(
            filters=filters,
            kernel_size=4,
            strides=2
        )

        self.attention0 = layers.Attention(
            use_scale=True,
            causal=False
        )

        self.avg = layers.GlobalAveragePooling1D()

        self.dense_output = layers.Dense(1, activation="sigmoid")

    def call(self, inputs, training=None, mask=None):
        q = self.query_mat(inputs)
        v = self.value_mat(inputs)

        x = self.attention0([q, v])
        x = self.avg(x)

        return self.dense_output(x)

    def model(self):
        x = layers.Input(shape=(500, 5))
        return tf.keras.Model(inputs=[x], outputs=self.call(x))