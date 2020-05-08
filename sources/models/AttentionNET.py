import tensorflow as tf
import tensorflow.keras.layers as layers

import config
from models.Blocks import *


class AttentionNET(tf.keras.Model):

    def __init__(self, hparams):
        super(AttentionNET, self).__init__()

        self.downres0 = DownResLayer(
            channels_out=4,
            dropout_rate=0.5,
            kernel_size=5,
            normalization="layer",
            first_layer=True,
            regularization=tf.keras.regularizers.l2(0.0005)
        )

        self.attention_layer = AttentionLayer(
            channels_out=4,
            filters_per_head=3,
            num_attention_heads=2,
            kernel_size=5,
            use_positional_encoding=True,
            regularization=tf.keras.regularizers.l2(0.0005)
        )

        self.downres1 = DownResLayer(
            channels_out=5,
            dropout_rate=0.25,
            kernel_size=6,
            normalization="layer",
            downsample_rate=3,
            regularization=tf.keras.regularizers.l2(0.0005)
        )

        self.downres_2 = DownResLayer(
            channels_out=6,
            dropout_rate=0.0,
            kernel_size=5,
            normalization="layer",
            downsample_rate=3,
            regularization=tf.keras.regularizers.l2(0.0005)
        )

        self.downres_3 = DownResLayer(
            channels_out=6,
            dropout_rate=0.0,
            kernel_size=5,
            normalization="layer",
            downsample_rate=3
        )

        self.avg = layers.GlobalAveragePooling1D()

        self.dense_output = layers.Dense(1, activation="sigmoid")

    def call(self, inputs, training=None, mask=None):
        x = self.downres0(inputs)
        x = self.attention_layer(x)
        x = self.downres1(x)
        x = self.downres_2(x)
        x = self.downres_3(x)
        x = self.avg(x)
        return self.dense_output(x)

    def model(self):
        x = layers.Input(shape=(500, 5), batch_size=128)
        return tf.keras.Model(inputs=[x], outputs=self.call(x))