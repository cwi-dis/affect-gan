import config

import tensorflow as tf
import tensorflow.keras.layers as layers

class ChannelCNN(tf.keras.Model):

    def __init__(self, hparams, n_channels):
        super(ChannelCNN, self).__init__()
        self.n_channels = n_channels

        self.channel_conv = [
            layers.Conv1D(
                filters=hparams[config.HP_CHANNEL_FILTERS],
                kernel_size=hparams[config.HP_CHANNEL_KERNEL],
                strides=(hparams[config.HP_CHANNEL_KERNEL] + 1) // 2,
                padding="valid",
                activation=layers.LeakyReLU()
            ) for c in tf.range(self.n_channels)
        ]

        self.channel_drop = [layers.Dropout(rate=0.4) for c in tf.range(self.n_channels)]

        self.channel_merge = layers.Conv1D(
            filters=1,
            kernel_size=hparams[config.HP_CHANNEL_MERGE_KERNEL],
            strides=(hparams[config.HP_CHANNEL_MERGE_KERNEL] + 1) // 2,
            activation=layers.LeakyReLU()
        )
        self.dense_out = layers.Dense(units=1, activation="softmax")




    def call(self, inputs, training=None, mask=None):
        x_c = tf.split(inputs, self.n_channels, -1)
        x_c1 = [self.channel_conv[c](x_c[c]) for c in tf.range(self.n_channels)]
        x_c2 = [self.channel_drop[c](x_c1[c], training=training) for c in tf.range(self.n_channels)]
        x = layers.concatenate(x_c2)
        x = self.channel_merge(x)
        return self.dense_out(x)


    def model(self):
        x = layers.Input(shape=(500, 5))
        return tf.keras.Model(inputs=[x], outputs=self.call(x))
