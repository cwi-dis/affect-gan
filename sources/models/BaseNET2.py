import config

import tensorflow as tf
import tensorflow.keras.layers as layers
from models.Blocks import *

class BaseNET2(tf.keras.Model):

    def __init__(self, hparams):
        super(BaseNET2, self).__init__()
        self.expand = layers.Conv1D(
            filters=4,
            kernel_size=5,
            padding="same"
        )

        self.downres0 = DownResLayer(
            channels_out=6,
            dropout_rate=0.0,
            kernel_size=6,
            first_layer=False,
            regularization=tf.keras.regularizers.l2(0.0005)
        )

        self.conv3 = layers.Conv1D(
            filters=8,
            kernel_size=6,
            strides=1,
            padding="same",
            activation="relu",
            kernel_regularizer=tf.keras.regularizers.l2(0.0005)
        )
        self.maxp = layers.MaxPool1D(2,2)

        self.flat = layers.Flatten()
        self.drop1 = layers.Dropout(0.5)
        self.dense_out = layers.Dense(2, activation="softmax")

    def call(self, inputs, training=None, mask=None):

        x = self.expand(inputs)
        x = self.downres0(x, training=training)
        x = self.conv3(x)
        x = self.maxp(x)
        x = self.flat(x)
        x = self.drop1(x, training=training)
        return self.dense_out(x)

    def model(self):
        x = layers.Input(shape=(500, 2))
        return tf.keras.Model(inputs=[x], outputs=self.call(x))
