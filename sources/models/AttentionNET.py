import tensorflow as tf
import tensorflow.keras.layers as layers

import config
from models.Blocks import *


class AttentionNET(tf.keras.Model):

    def __init__(self, hparams):
        super(AttentionNET, self).__init__()

        self.downres0 = DownResLayer(
            channels_out=6,
            dropout_rate=0.2,
            kernel_size=6,
            first_layer=True,
            regularization=tf.keras.regularizers.l2(0.0005)
        )

        self.max_pool0 = layers.MaxPool1D(pool_size=2, strides=2)
        self.attention_layer = AttentionLayer(
            channels_out=6,
            filters_per_head=4,
            num_attention_heads=1,
            use_positional_encoding=False,
            regularization=tf.keras.regularizers.l2(0.0005),
            kernel_size=6,
        )

        self.max_pool1 = layers.MaxPool1D(pool_size=2, strides=2)

        self.downres1 = DownResLayer(
            channels_out=8,
            dropout_rate=0.2,
            kernel_size=6,
            regularization=tf.keras.regularizers.l2(0.0005)
        )

        self.final_pool = layers.MaxPool1D(pool_size=2, strides=2)
        self.dense_dropout = layers.Dropout(0.5)

        self.dense_output = layers.Dense(2, activation="softmax")

    def call(self, inputs, training=None, mask=None):
        x = self.downres0(inputs)
        x = self.max_pool0(x)
        x = self.attention_layer(x)
        x = self.max_pool1(x)
        x = self.downres1(x)
        x = self.final_pool(x)

        return self.dense_output(x)

    def model(self):
        x = layers.Input(shape=(500, 2), batch_size=128)
        return tf.keras.Model(inputs=[x], outputs=self.call(x))
