import config

import tensorflow as tf
import tensorflow.keras.layers as layers
from models.Blocks import *

class BaseNET2(tf.keras.Model):

    def __init__(self, hparams):
        super(BaseNET2, self).__init__()

        self.downres0 = DownResLayer(
            channels_out=4,
            dropout_rate=0.25,
            kernel_size=6,
            normalization="batch",
            first_layer=True,
            regularization=tf.keras.regularizers.l2(0.0005)
        )

        self.downres1 = DownResLayer(
            channels_out=4*2,
            dropout_rate=0.25,
            kernel_size=6,
            normalization="batch",
            regularization=tf.keras.regularizers.l2(0.0005)
        )

        self.max0 = layers.MaxPool1D(
            pool_size=2,
            strides=2,
            padding="same"
        )
        self.drop1 = layers.Dropout(0.5)
        self.flat = layers.Flatten()
        self.dense_out = layers.Dense(2, activation="softmax")

    def call(self, inputs, training=None, mask=None):

        x = self.downres0(inputs, training=training)
        x = self.downres1(x, training=training)
        x = self.max0(x)

        x = self.flat(x)
        x = self.drop1(x, training)

        return self.dense_out(x)

    def model(self):
        x = layers.Input(shape=(500, 2))
        return tf.keras.Model(inputs=[x], outputs=self.call(x))
