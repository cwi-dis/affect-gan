import config

import tensorflow as tf
import tensorflow.keras.layers as layers


class BaseNET1(tf.keras.Model):

    def __init__(self, hparams):
        super(BaseNET1, self).__init__()
        #self.dense_shape = tf.math.ceil(config.INPUT_SIZE / hparams[config.HP_POOL]) * hparams[config.HP_FILTERS]

        self.drop0 = layers.Dropout(0.1)
        self.conv0 = layers.Conv1D(
            filters=hparams[config.HP_FILTERS],
            kernel_size=8,
            strides=2,
            padding="same",
            activation=layers.LeakyReLU())
        self.max0 = layers.MaxPool1D(
            pool_size=2,
            strides=2,
            padding="same"
        )
        self.drop1 = layers.Dropout(0.2)
        self.conv1 = layers.Conv1D(
            filters=hparams[config.HP_FILTERS] * 2,
            kernel_size=6,
            strides=2,
            padding="same",
            activation=layers.LeakyReLU()
        )
        self.max1 = layers.MaxPool1D(
            pool_size=2,
            strides=2,
            padding="same"
        )
        self.drop2 = layers.Dropout(0.2)
        self.flat = layers.Flatten()
        self.dense_out = layers.Dense(2, activation="softmax")

    def call(self, inputs, training=None, mask=None):

        x = self.drop0(inputs, training)
        x = self.conv0(x)
        x = self.max0(x)

        x = self.drop1(x, training)
        x = self.conv1(x)
        x = self.max1(x)

        x = self.flat(x)
        x = self.drop2(x, training)

        return self.dense_out(x)

    def model(self):
        x = layers.Input(shape=(500, 2))
        return tf.keras.Model(inputs=[x], outputs=self.call(x))
