import tensorflow as tf
import tensorflow.keras.layers as layers

import config
from models.Blocks import *


class Generator(tf.keras.Model):

    def __init__(self, n_signals, *args, **kwargs):
        super(Generator, self).__init__(*args, **kwargs)
        self.n_multiplier = 2
        self.expand = layers.Dense(units=125 * 40, use_bias=False)
        self.up_0 = UpResLayer(channels_out=40, kernel_size=6, dropout_rate=0.2, normalization=None)
        self.non_local = AttentionLayer(
            name="att0",
            channels_out=40,
            kernel_size=6,
            filters_per_head=10,
            num_attention_heads=4,
            use_positional_encoding=False)
        self.up_1 = UpResLayer(channels_out=20, kernel_size=8, dropout_rate=0.2, normalization=None)
        self.act = layers.LeakyReLU(alpha=0.2)
        self.final_conv = layers.Conv1D(filters=n_signals, kernel_size=10, padding="same")

    def call(self, inputs, training=None, mask=None):
        x = self.expand(inputs)
        x = tf.reshape(x, shape=[-1, 125, 40])
        x = self.up_0(x, training=training)
        x = self.non_local(x)
        x = self.act(self.up_1(x, training=training))
        x = self.final_conv(x)
        # x = tf.keras.activations.tanh(x)
        return x

    def model(self):
        x = layers.Input(shape=(125), batch_size=2)
        return tf.keras.Model(inputs=[x], outputs=self.call(x))


class Discriminator(tf.keras.Model):
    def __init__(self, class_conditional, subject_conditional, *args, **kwargs):
        super(Discriminator, self).__init__(*args, **kwargs)
        self.class_conditional = class_conditional
        self.subject_conditional = subject_conditional
        self.out_channels = 36

        self.expand = layers.Conv1D(filters=self.out_channels // 4, kernel_size=5, padding="same")
        self.downres0 = DownResLayer(
            channels_out=self.out_channels // 3,
            dropout_rate=0.4,
            kernel_size=6,
            first_layer=True,
            normalization="layer"
        )
        self.non_local = AttentionLayer(
            name="att0",
            channels_out=self.out_channels // 3,
            filters_per_head=8,
            num_attention_heads=3,
            kernel_size=5,
            use_positional_encoding=True
        )
        self.downres1 = DownResLayer(
            channels_out=self.out_channels // 2,
            kernel_size=6,
            dropout_rate=0.25,
            normalization="layer"
        )
        self.downres2 = DownResLayer(
            channels_out=self.out_channels,
            kernel_size=6,
            dropout_rate=0.0,
            normalization=None
        )

        self.dense = layers.Dense(256, activation=layers.LeakyReLU(0.2))

        self.dense_output = layers.Dense(1)
        self.dense_class_output = layers.Dense(2, activation="softmax")
        self.dense_subject_output = layers.Dense(29, activation="softmax")

    def call(self, inputs, training=None, mask=None):
        x = self.expand(inputs)
        x = self.downres0(x, training=training)
        x = self.non_local(x, training=training)
        x = self.downres1(x, training=training)
        x = self.downres2(x, training=training)

        x_s = tf.reshape(x, shape=[-1, 63 * self.out_channels])
        x_s = self.dense(x_s)

        c = 0
        s = 0
        if self.class_conditional:
            c = self.dense_class_output(x_s)

        if self.subject_conditional:
            s = self.dense_subject_output(x_s)

        x_s = self.dense_output(x_s)

        return tf.keras.activations.sigmoid(x_s), x_s, c, s

    def model(self):
        x = layers.Input(shape=(500, 5), batch_size=2)
        return tf.keras.Model(inputs=[x], outputs=self.call(x))
