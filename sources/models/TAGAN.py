import tensorflow as tf
import tensorflow.keras.layers as layers

import config
from models.Blocks import *

class Generator(tf.keras.Model):

    def __init__(self, n_signals, batch_size, *args, **kwargs):
        super(Generator, self).__init__(*args, **kwargs)
        
        self.expand = layers.Dense(units=125 * 20, use_bias=False)
        self.up_0 = UpResLayer(channels_out=20, kernel_size=5, dropout_rate=0.3, normalization="layer")
        self.up_1 = UpResLayer(channels_out=10, kernel_size=7, dropout_rate=0.0, normalization="layer")
        self.non_local = AttentionLayer(
            channels_out=20,
            kernel_size=3,
            filters_per_head=5,
            num_attention_heads=4,
            use_positional_encoding=True,
            batch_size=batch_size)
        self.act = layers.LeakyReLU(alpha=0.2)
        self.final_conv = layers.Conv1D(filters=n_signals, kernel_size=7, padding="same")

    def call(self, inputs, training=None, mask=None):
        x = self.expand(inputs)
        x = tf.reshape(x, shape=[-1, 125, 20])
        x = self.up_0(x, training=training)
        x = self.non_local(x)
        x = self.up_1(x, training=training)
        x = self.final_conv(x)
        x = tf.keras.activations.tanh(x)
        return x

    def model(self):
        x = layers.Input(shape=(125), batch_size=2)
        return tf.keras.Model(inputs=[x], outputs=self.call(x))


class Discriminator(tf.keras.Model):
    def __init__(self, conditional, batch_size, *args, **kwargs):
        super(Discriminator, self).__init__(*args, **kwargs)
        self.conditional = conditional
        self.out_channels = 24
        self.downres0 = DownResLayer(
            channels_out=self.out_channels // 3,
            dropout_rate=0.3,
            kernel_size=5,
            first_layer=True,
            normalization="layer"
        )
        self.non_local = AttentionLayer(
            channels_out=self.out_channels // 3,
            filters_per_head=4,
            num_attention_heads=4,
            kernel_size=3,
            use_positional_encoding=True,
            batch_size=batch_size
        )
        self.downres1 = DownResLayer(
            channels_out=self.out_channels // 2,
            kernel_size=5,
            dropout_rate=0.0,
            normalization="layer"
        )
        self.downres2 = DownResLayer(
            channels_out=self.out_channels,
            kernel_size=5,
            dropout_rate=0.0,
            normalization="layer"
        )

        self.avg = layers.GlobalAveragePooling1D()

        self.dense_output = layers.Dense(1)

        self.dense_class_output = layers.Dense(1, activation="sigmoid")

    def call(self, inputs, training=None, mask=None):
        x = self.downres0(inputs, training=training)
        x = self.non_local(x, training=training)
        x = self.downres1(x, training=training)
        x = self.downres2(x, training=training)

        x_s = tf.reshape(x, shape=[-1, 63*self.out_channels])
        c = None
        if self.conditional:
            #c = self.avg(x)
            c = self.dense_class_output(x_s)

        x_s = self.dense_output(x_s)
        return tf.keras.activations.sigmoid(x), x_s, c

    def model(self):
        x = layers.Input(shape=(500, 1), batch_size=2)
        return tf.keras.Model(inputs=[x], outputs=self.call(x))