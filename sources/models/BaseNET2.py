import config

import tensorflow as tf
import tensorflow.keras.layers as layers
from models.Blocks import *

class BaseNET2(tf.keras.Model):

    def __init__(self, hparams):
        super(BaseNET2, self).__init__()

        self.expand = layers.Conv1D(
            filters=6,
            kernel_size=5,
            padding="same"
        )

        self.downres0 = DownResLayer(
            channels_out=8,
            dropout_rate=0.0,
            kernel_size=6,
            first_layer=False,
            regularization=tf.keras.regularizers.l2(0.0005)
        )

        self.conv3 = layers.Conv1D(
            filters=10,
            kernel_size=25,
            strides=5,
            padding="same",
            kernel_regularizer=tf.keras.regularizers.l2(0.0005)
        )

        self.flat = layers.Flatten()
        self.drop1 = layers.Dropout(0.5)
        self.dense_out = layers.Dense(2, activation="softmax")

    def call(self, inputs, training=None, mask=None):

        x = self.expand(inputs)
        x = self.downres0(x, training=training)
        x = self.conv3(x)

        x = self.flat(x)
        x = self.drop1(x, training)

        return self.dense_out(x)

    def model(self):
        x = layers.Input(shape=(500, 2))
        return tf.keras.Model(inputs=[x], outputs=self.call(x))
