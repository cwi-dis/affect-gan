import tensorflow as tf
import tensorflow.keras.layers as layers

import config
from models.Blocks import *

class Generator(tf.keras.Model):

    def __init__(self, n_signals, *args, **kwargs):
        super(Generator, self).__init__(*args, **kwargs)
        
        self.expand = layers.Dense(units=125 * 32, use_bias=False)
        self.up_0 = UpResLayer(channels_out=16, kernel_size=4, dropout_rate=0.0)
        self.non_local = AttentionLayer(channels_out=16, filters=4)
        self.up_1 = UpResLayer(channels_out=8, kernel_size=4, dropout_rate=0.0)
        self.act = layers.LeakyReLU(alpha=0.2)
        self.final_conv = layers.Conv1D(filters=n_signals, kernel_size=3, padding="same")

    def call(self, inputs, training=None, mask=None):
        x = self.expand(inputs)
        x = tf.reshape(x, shape=[-1, 125, 32])
        x = self.up_0(x, training=training)
        x = self.non_local(x)
        x = self.act(self.up_1(x, training=training))
        x = self.final_conv(x)
        x = tf.keras.activations.tanh(x)
        return x

    def model(self):
        x = layers.Input(shape=(125))
        return tf.keras.Model(inputs=[x], outputs=self.call(x))


class Discriminator(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super(Discriminator, self).__init__(*args, **kwargs)

        self.downres0 = DownResLayer(
            channels_out=4,
            dropout_rate=0.5,
            first_layer=True,
            use_dropout=True
        )
        self.non_local = AttentionLayer(
            channels_out=4,
            filters=2
        )
        self.downres1 = DownResLayer(
            channels_out=6,
            use_dropout=False
        )
        self.downres2 = DownResLayer(
            channels_out=8,
            use_dropout=False
        )

        self.avg = layers.GlobalAveragePooling1D()

        self.dense_output = layers.Dense(1)

    def call(self, inputs, training=None, mask=None):
        x = self.downres0(inputs, training=training)
        x = self.non_local(x, training=training)
        x = self.downres1(x, training=training)
        x = self.downres2(x, training=training)

        x = self.avg(x)
        x = self.dense_output
        return tf.sigmoid(x), x

    def model(self):
        x = layers.Input(shape=(500, 1))
        return tf.keras.Model(inputs=[x], outputs=self.call(x))