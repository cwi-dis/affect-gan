import config

import tensorflow as tf
import tensorflow.keras.layers as layers


class BaseNET1(tf.keras.Model):

    def __init__(self, hparams):
        super(BaseNET1, self).__init__()
        self.dense_shape = tf.math.ceil(config.INPUT_SIZE / hparams[config.HP_POOL]) * hparams[config.HP_FILTERS]

        self.conv_1 = layers.Conv1D(
            filters=hparams[config.HP_FILTERS],
            kernel_size=hparams[config.HP_KERNEL],
            dilation_rate=hparams[config.HP_DILATION],
            padding="same", activation=layers.LeakyReLU())
        self.drop = layers.Dropout(rate=hparams[config.HP_DROPOUT])
        self.avg = layers.AveragePooling1D(
            pool_size=hparams[config.HP_POOL],
            strides=hparams[config.HP_POOL],
            padding="same")
        self.flat = layers.Flatten()
        self.dense = layers.Dense(8, input_shape=(self.dense_shape,), activation=layers.LeakyReLU())
        self.dense_out = layers.Dense(1)

    def call(self, inputs, training=None, mask=None):

        x = self.conv_1(inputs)
        x = self.drop(x, training)
        x = self.avg(x)
        x = self.flat(x)
        #x = tf.reshape(x, [-1, ])
        #x = self.dense(x)
        return self.dense_out(x)

    def model(self):
        x = layers.Input(shape=(500, 5))
        return tf.keras.Model(inputs=[x], outputs=self.call(x))
